import torch
from torch import cuda
import triton
import triton.language as tl
import pdb

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_autotune_config():
    if is_cuda():
        return [
        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
            for BM in [4, 8, 16] \
            for BN in [4, 8, 16] \
            for s in [2, 4, 8] \
            for w in [4, 8, 16]
    ]
    else:
        raise NotImplementedError("Not support yet.")

@triton.autotune(
    configs=get_autotune_config(),
    key=['Z', 'S', 'H', 'N_CTX', 'N_COORDS', 'HEAD_DIM'],
)
@triton.jit
def _fwd_kernel(
    Q, K, V, COORD, sm_scale,
    M,
    Out,
    stride_qz, stride_qs, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_ks, stride_kh, stride_km, stride_kk,
    stride_vz, stride_vs, stride_vh, stride_vm, stride_vk,
    stride_oz, stride_os, stride_oh, stride_om, stride_ok,
    stride_iz, stride_is0, stride_im, stride_is1, stride_ip, stride_ic,
    stride_mz, stride_ms, stride_mh, stride_ml,
    Z, S, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr, HEIGHT: tl.constexpr, 
    N_COORDS: tl.constexpr,
    DTYPE_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_s = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_s.to(tl.int64) * stride_qs + off_h.to(tl.int64) * stride_qh
    o_offset = off_z.to(tl.int64) * stride_oz + off_s.to(tl.int64) * stride_os + off_h.to(tl.int64) * stride_oh
    coord_offset = off_z.to(tl.int64) * stride_iz + off_s.to(tl.int64) * stride_is0
    row_mask = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < N_CTX
    
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    # COORD_block_ptr = tl.make_block_ptr(
    #     base=COORD + coord_offset,
    #     shape=(N_CTX, S * N_COORDS, 2),
    #     strides=(stride_im, stride_ip, stride_ic),
    #     offsets=(start_m * BLOCK_M, 0, 0),
    #     block_shape=(BLOCK_M, BLOCK_N, 2),
    #     order=(2,1,0)
    # )
    COORDX_block_ptr = tl.make_block_ptr(
        base        = COORD + coord_offset,
        shape       = (N_CTX, S * N_COORDS),
        strides     = (stride_im, stride_ip),
        offsets     = (start_m * BLOCK_M, 0),
        block_shape = (BLOCK_M, BLOCK_N),
        order       = (1, 0)
    )

    COORDY_block_ptr = tl.make_block_ptr(
        base        = COORD + coord_offset + stride_ic,
        shape       = (N_CTX, S * N_COORDS),
        strides     = (stride_im, stride_ip),
        offsets     = (start_m * BLOCK_M, 0),
        block_shape = (BLOCK_M, BLOCK_N),
        order       = (1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kvzh = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    
    K_ptr_base = K + offs_kvzh + tl.arange(0, HEAD_DIM) * stride_kk
    V_ptr_base = V + offs_kvzh + tl.arange(0, HEAD_DIM) * stride_vk
        
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # online maximum
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0  # online denominator
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32) # online output
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,))
    q = tl.where(row_mask[:, None], q, 0.0)
    q = (q.to(tl.float32) * qk_scale)
    
    hi = S * N_COORDS
    # loop over k, v and update accumulator
    for start_n in tl.range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load coords --
        # coords = tl.load(COORD_block_ptr) # M, N, 2
        # cols, rows = coords[:, :, 0], coords[:, :, 1] # [BLOCK_M, BLOCK_N]
        cols = tl.load(COORDX_block_ptr, boundary_check=(0,1)).to(tl.float32) # [BLOCK_M, BLOCK_N]
        rows = tl.load(COORDY_block_ptr, boundary_check=(0,1)).to(tl.float32) # [BLOCK_M, BLOCK_N]
        s_offset = (start_n + tl.arange(0, BLOCK_N)) // N_COORDS
        # -- load kv -- from Z, S, H, L, D
        K_ptr = K_ptr_base[None, :] + s_offset[:, None] * stride_ks # [N(S), D]
        V_ptr = V_ptr_base[None, :] + s_offset[:, None] * stride_vs # [N(S), D]

        mask = row_mask[:, None] & ((start_n + tl.arange(0, BLOCK_N)) < hi)[None,:]
        coord_mask = mask & (cols[:,:] < WIDTH) & (rows[:,:] < HEIGHT) & (cols[:,:] >=0) & (rows[:,:] >=0)
        
        # TODO: check the grid sample

        col0  = tl.floor(cols).to(tl.int32)
        row0  = tl.floor(rows).to(tl.int32)
        col1  = tl.minimum(col0 + 1, WIDTH - 1)
        row1  = tl.minimum(row0 + 1, HEIGHT - 1)
        pos00 = (row0 * WIDTH + col0).to(tl.int32)
        pos01 = (row0 * WIDTH + col1).to(tl.int32)
        pos10 = (row1 * WIDTH + col0).to(tl.int32)
        pos11 = (row1 * WIDTH + col1).to(tl.int32)
                
        k00 = tl.load(K_ptr[None, :, :] + pos00[:, :, None] * stride_km, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v00 = tl.load(V_ptr[None, :, :] + pos00[:, :, None] * stride_vm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        k01 = tl.load(K_ptr[None, :, :] + pos01[:, :, None] * stride_km, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v01 = tl.load(V_ptr[None, :, :] + pos01[:, :, None] * stride_vm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        k10 = tl.load(K_ptr[None, :, :] + pos10[:, :, None] * stride_km, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v10 = tl.load(V_ptr[None, :, :] + pos10[:, :, None] * stride_vm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        k11 = tl.load(K_ptr[None, :, :] + pos11[:, :, None] * stride_km, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v11 = tl.load(V_ptr[None, :, :] + pos11[:, :, None] * stride_vm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        
        w_col1 = cols - col0
        w_col0 = 1.0 - w_col1
        w_row1 = rows - row0
        w_row0 = 1.0 - w_row1
        w00 = (w_col0 * w_row0)[:, :, None] # left-top
        w01 = (w_col1 * w_row0)[:, :, None] # right-top
        w10 = (w_col0 * w_row1)[:, :, None] # left-bottom
        w11 = (w_col1 * w_row1)[:, :, None] # right-bottom
        # k = (w00*k00 + w01*k01 + w10*k10 + w11*k11) # [M, N, D]
        # v = (w00*v00 + w01*v01 + w10*v10 + w11*v11) # [M, N, D]
        k = tl.fma(w00, k00, tl.fma(w01, k01, tl.fma(w10, k10, w11 * k11)))  # [M,N,D], fp32
        v = tl.fma(w00, v00, tl.fma(w01, v01, tl.fma(w10, v10, w11 * v11)))  # [M,N,D], fp32
                
        # -- compute qk ----
        qk = tl.sum(q[:, None, :] * k, axis=2)
        qk = tl.where(mask, qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        p = tl.where(mask, p, 0.0)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        acc = acc + tl.sum(p[:,:,None]*v, axis=1)
        # update m_i and l_i
        m_i = m_ij
        
        # COORD_block_ptr = tl.advance(COORD_block_ptr, (0, BLOCK_N, 0))
        COORDX_block_ptr = tl.advance(COORDX_block_ptr, (0, BLOCK_N))
        COORDY_block_ptr = tl.advance(COORDY_block_ptr, (0, BLOCK_N))
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_z.to(tl.int64) * stride_mz + off_s.to(tl.int64) * stride_ms + off_h.to(tl.int64) * stride_mh + offs_m * stride_ml
    tl.store(m_ptrs, m_i, mask=row_mask)
    if DTYPE_MODE == 0: # branch is controlled by compliler
        tl.store(O_block_ptr, acc, boundary_check=(0,))
    else:
        tl.store(O_block_ptr, acc.to(tl.float16), boundary_check=(0,))

@triton.jit
def _bwd_preprocess(o_ptr, do_ptr, delta_ptr,
                         z, S, H, N_CTX,
                         stride_oz, stride_os, stride_oh, stride_om, stride_ok,
                         stride_deltaz, stride_deltas, stride_deltah, stride_deltam,
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr):
    """Compute delta = sum(dOut * Out) for each query (for softmax backward)."""
    off_hz = tl.program_id(1)
    off_s = tl.program_id(2)
    off_z = off_hz // H
    off_h = off_hz % H
    off_n = tl.arange(0, HEAD_DIM)
    
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    
    o_offset = off_z.to(tl.int64) * stride_oz + off_s.to(tl.int64) * stride_os + off_h.to(tl.int64) * stride_oh + off_m[:, None] * stride_om + off_n[None, :]
    o = tl.load(o_ptr + o_offset, mask=(off_m < N_CTX)[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + o_offset, mask=(off_m < N_CTX)[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=-1)
    tl.store(delta_ptr + off_z.to(tl.int64) * stride_deltaz + off_s.to(tl.int64) * stride_deltas + off_h.to(tl.int64) * stride_deltah + off_m * stride_deltam, delta, mask=off_m < N_CTX)


@triton.jit
def _bwd_kernel(Q, K, V, COORD, sm_scale,
              DO, DQ, DK, DV,
              M, D,
              # shared by Q/K/V/DO.
              stride_qz, stride_qs, stride_qh, stride_qm, stride_qk,
              stride_iz, stride_is0, stride_im, stride_is1, stride_ip, stride_ic,
              stride_mz, stride_ms, stride_mh, stride_ml,
              Z, S, H, N_CTX,
              HEAD_DIM: tl.constexpr,
              WIDTH: tl.constexpr, HEIGHT: tl.constexpr, 
              DTYPE_MODE: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              N_COORDS: tl.constexpr,
):  
    LN2: tl.constexpr = 0.6931471805599453 # = ln(2)

    pid = tl.program_id(0)
    bhid = tl.program_id(1)
    off_s = tl.program_id(2)
    off_z = bhid // H
    off_h = bhid % H
    
    offs_d = tl.arange(0, HEAD_DIM)
    start_m = pid * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    row_mask = offs_m < N_CTX
    
    kv_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh + offs_d * stride_qk
    q_offset = kv_offset + off_s.to(tl.int64) * stride_qs
    coord_offset = off_z.to(tl.int64) * stride_iz + off_s.to(tl.int64) * stride_is0
    
    # offset pointers for batch/head
    Q += q_offset
    K += kv_offset# Be careful about KV
    V += kv_offset
    DO += q_offset
    DQ += q_offset
    DK += kv_offset
    DV += kv_offset
    COORD += coord_offset
    m_offset = off_z.to(tl.int64) * stride_mz + off_s.to(tl.int64) * stride_ms \
        + off_h.to(tl.int64) * stride_mh + offs_m * stride_ml
    
    DQ_ptr = DQ[None, :] + offs_m[:, None] * stride_qm
    q = tl.load(
        Q[None, :] + offs_m[:, None] * stride_qm,
        mask = row_mask[:, None], other=0.
    ).to(tl.float32) # M, D
    do = tl.load(
        DO[None, :] + offs_m[:, None] * stride_qm,
        mask = row_mask[:, None], other=0.
    ).to(tl.float32)
    M_i = tl.load(M + m_offset, mask=row_mask, other=0.) # M
    D_i = tl.load(D + m_offset, mask=row_mask, other=0.) # M

    hi = S * N_COORDS

    dq_block = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in tl.range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        mask = row_mask[:, None] & (offs_n < hi)[None,:]
        
        cols = tl.load(COORD + offs_m[:, None] * stride_im + 
                       offs_n[None, :] * stride_ip + 0 * stride_ic,
                       mask = mask
                       ).to(tl.float32) # [BLOCK_M, BLOCK_N]
        rows = tl.load(COORD + offs_m[:, None] * stride_im + 
                       offs_n[None, :] * stride_ip + 1 * stride_ic,
                       mask = mask
                       ).to(tl.float32) # [BLOCK_M, BLOCK_N]
        coord_mask = mask & (cols < WIDTH) & (rows < HEIGHT) & (cols >=0) & (rows >=0)
        
        col0  = tl.floor(cols).to(tl.int32)
        row0  = tl.floor(rows).to(tl.int32)
        col1  = tl.minimum(col0 + 1, WIDTH - 1)
        row1  = tl.minimum(row0 + 1, HEIGHT - 1)
        pos00 = (row0 * WIDTH + col0).to(tl.int32)
        pos01 = (row0 * WIDTH + col1).to(tl.int32)
        pos10 = (row1 * WIDTH + col0).to(tl.int32)
        pos11 = (row1 * WIDTH + col1).to(tl.int32)
        
        s_offset = (start_n + tl.arange(0, BLOCK_N)) // N_COORDS
        K_ptr = K[None, :] + s_offset[:, None] * stride_qs # [N(S), D]
        V_ptr = V[None, :] + s_offset[:, None] * stride_qs # [N(S), D]
                
        k00 = tl.load(K_ptr[None, :, :] + pos00[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v00 = tl.load(V_ptr[None, :, :] + pos00[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        k01 = tl.load(K_ptr[None, :, :] + pos01[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v01 = tl.load(V_ptr[None, :, :] + pos01[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        k10 = tl.load(K_ptr[None, :, :] + pos10[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v10 = tl.load(V_ptr[None, :, :] + pos10[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        k11 = tl.load(K_ptr[None, :, :] + pos11[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        v11 = tl.load(V_ptr[None, :, :] + pos11[:, :, None] * stride_qm, mask=coord_mask[:,:,None], other=0.0).to(tl.float32) # [M, P, D]
        
        w_col1 = cols - col0
        w_col0 = 1.0 - w_col1
        w_row1 = rows - row0
        w_row0 = 1.0 - w_row1
        w00 = (w_col0 * w_row0)[:, :, None] # left-top
        w01 = (w_col1 * w_row0)[:, :, None] # right-top
        w10 = (w_col0 * w_row1)[:, :, None] # left-bottom
        w11 = (w_col1 * w_row1)[:, :, None] # right-bottom
        
        k_interp = w00*k00 + w01*k01 + w10*k10 + w11*k11 # [M, N, D]
        v_interp = w00*v00 + w01*v01 + w10*v10 + w11*v11 # [M, N, D]
        # k_interp = tl.fma(w00, k00, tl.fma(w01, k01, tl.fma(w10, k10, w11 * k11)))  # [M, N, D], fp32
        # v_interp = tl.fma(w00, v00, tl.fma(w01, v01, tl.fma(w10, v10, w11 * v11)))  # [M, N, D], fp32

        qk = tl.sum(q[:, None, :] * k_interp, axis=2) # [BLOCK_M, BLOCK_N]
        qk = qk - M_i[:, None]
        p = tl.math.exp2(qk)
        p = tl.where(mask, p, 0.0)
        
        dp = tl.sum(do[:, None, :] * v_interp, axis=2) 
        dp = dp - D_i[:, None]
        ds = p * dp # [BLOCK_M, BLOCK_N]
        
        dq_block += tl.sum(ds[:,:,None] * k_interp * LN2, axis=1)
        
        # TODO: reduce atomic conflict
        DK_ptr = DK[None, :] + s_offset[:, None] * stride_qs # [N(S), D]
        DV_ptr = DV[None, :] + s_offset[:, None] * stride_qs # [N(S), D]
        
        grad_k00 = ds[:,:,None] * q[:,None,:] * w00 * sm_scale
        tl.atomic_add(DK_ptr[None, :, :] + pos00[:, :, None] * stride_qm, grad_k00, mask=coord_mask[:,:,None])
        grad_v00 = p[:,:,None] * do[:,None,:] * w00
        tl.atomic_add(DV_ptr[None, :, :] + pos00[:, :, None] * stride_qm, grad_v00, mask=coord_mask[:,:,None])
        grad_k01 = ds[:,:,None] * q[:,None,:] * w01 * sm_scale
        tl.atomic_add(DK_ptr[None, :, :] + pos01[:, :, None] * stride_qm, grad_k01, mask=coord_mask[:,:,None])
        grad_v01 = p[:,:,None] * do[:,None,:] * w01
        tl.atomic_add(DV_ptr[None, :, :] + pos01[:, :, None] * stride_qm, grad_v01, mask=coord_mask[:,:,None])
        grad_k10 = ds[:,:,None] * q[:,None,:] * w10 * sm_scale
        tl.atomic_add(DK_ptr[None, :, :] + pos10[:, :, None] * stride_qm, grad_k10, mask=coord_mask[:,:,None])
        grad_v10 = p[:,:,None] * do[:,None,:] * w10
        tl.atomic_add(DV_ptr[None, :, :] + pos10[:, :, None] * stride_qm, grad_v10, mask=coord_mask[:,:,None])
        grad_k11 = ds[:,:,None] * q[:,None,:] * w11 * sm_scale
        tl.atomic_add(DK_ptr[None, :, :] + pos11[:, :, None] * stride_qm, grad_k11, mask=coord_mask[:,:,None])
        grad_v11 = p[:,:,None] * do[:,None,:] * w11
        tl.atomic_add(DV_ptr[None, :, :] + pos11[:, :, None] * stride_qm, grad_v11, mask=coord_mask[:,:,None])
        
    if DTYPE_MODE == 0:
        tl.store(DQ_ptr, dq_block, mask=row_mask[:, None])
    else:
        tl.store(DQ_ptr, dq_block.to(tl.float16), mask=row_mask[:, None])

def _dtype_mode_from(t: torch.Tensor) -> int:
    if t.dtype == torch.float32:
        return 0
    elif t.dtype == torch.float16:
        return 1
    elif t.dtype == torch.bfloat16:
        return 1
    else:
        raise NotImplementedError

class _fused_deform_sdpa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, coords, WIDTH, HEIGHT, sm_scale):
        # TODO: merge k and v
        coords = coords.clone()
        coords[...,0] *= WIDTH - 1
        coords[...,1] *= HEIGHT - 1
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert q.shape[3] == WIDTH * HEIGHT
        assert coords.is_contiguous()
        o = torch.zeros_like(q)
        N_COORDS = coords.shape[-2]
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)
        # Z, S, H, L, D = q.shape
        grid = lambda META: (triton.cdiv(q.shape[3], META["BLOCK_M"]), q.shape[0] * q.shape[2], q.shape[1])
        ctx.grid = grid
        
        _fwd_kernel[grid](
            q, k, v, coords, sm_scale,
            M,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3), o.stride(4),
            coords.stride(0), coords.stride(1), coords.stride(2), coords.stride(3), coords.stride(4), coords.stride(5),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), 
            q.shape[0], q.shape[1], q.shape[2], q.shape[3],
            HEAD_DIM=HEAD_DIM_Q,
            WIDTH=WIDTH, HEIGHT=HEIGHT,
            N_COORDS=N_COORDS,
            DTYPE_MODE=_dtype_mode_from(q),
        )
        
        ctx.save_for_backward(q, k, v, coords, o, M)
        ctx.sm_scale = sm_scale
        ctx.WIDTH, ctx.HEIGHT = WIDTH, HEIGHT
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, coords, o, M = ctx.saved_tensors # here coords has been in-place denormlized
        WIDTH, HEIGHT = ctx.WIDTH, ctx.HEIGHT
        N_COORDS = coords.shape[-2]
        B, S, H, N_CTX, HEAD_DIM_Q = q.shape
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        assert do.is_contiguous()
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride() # TODO: reduce the # of arguments in fwd kernel
        dq = torch.zeros_like(q)
        dk32 = torch.zeros_like(k, dtype=torch.float32)
        dv32 = torch.zeros_like(v, dtype=torch.float32)
        delta = torch.zeros_like(M)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        PRE_BLOCK, BLOCK_M, BLOCK_N = 128, 8, 8
        NUM_WARPS, NUM_STAGES = 16, 2
        grid_pre = (triton.cdiv(N_CTX, PRE_BLOCK), B * H, S)
        _bwd_preprocess[grid_pre](o, do, delta,
                                       B, S, H, N_CTX,
                                       o.stride(0), o.stride(1), o.stride(2), o.stride(3), o.stride(4),
                                       delta.stride(0), delta.stride(1), delta.stride(2), delta.stride(3),
                                       BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM_Q)
        
        _bwd_kernel[ctx.grid](
            q, arg_k, v, coords, 
            ctx.sm_scale, 
            do, dq, dk32, dv32,
            M, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4),
            coords.stride(0), coords.stride(1), coords.stride(2), coords.stride(3), coords.stride(4), coords.stride(5),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), 
            B, S, H, N_CTX,
            HEAD_DIM=HEAD_DIM_Q,
            WIDTH=WIDTH, HEIGHT=HEIGHT,
            DTYPE_MODE=_dtype_mode_from(q),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            N_COORDS=N_COORDS,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES
        )

        dk = dk32.to(k.dtype)
        dv = dv32.to(v.dtype)
        
        return dq, dk, dv, None, None, None, None

deform_sdpa_triton = _fused_deform_sdpa.apply