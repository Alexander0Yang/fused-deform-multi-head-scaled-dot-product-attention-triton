import os
import argparse
import torch
import triton.testing as testing

from ops.triton import deform_sdpa_triton as fused_deform_sdpa
from ops.torch import deform_sdpa_torch as ref_deform_sdpa

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark script for Deformable Scaled Dot-Product Attention.")

    # Add arguments with types, default values, and help descriptions
    parser.add_argument('--b', type=int, default=2, help='Batch size.')
    parser.add_argument('--h', type=int, default=32, help='Number of attention heads.')
    parser.add_argument('--d', type=int, default=32, help='Dimension of each attention head.')
    parser.add_argument('--iw', type=int, default=18, help='Width of the image (height is set to be the same).')
    parser.add_argument('--sm-scale', type=float, default=0.5)

    args = parser.parse_args()
    
    BATCH = args.b
    N_HEADS = args.h
    HEAD_DIM = args.d
    W_IMG = args.iw
    SM_SCALE = args.sm_scale
    DTYPE = torch.float32
    S_LIST = [2,4,6,8,10,12]
    P_LIST = [2,4,6,8,10,12]
    SAVE_PATH = './results'
        
    configs = []
    for mode in ["fwd", "bwd"]:
        for P_samples in P_LIST:
            configs.append(
                testing.Benchmark(
                    x_names=["S_views"],
                    x_vals=S_LIST,
                    line_arg="provider",
                    line_vals=["triton-coord", "torch-ref"],
                    line_names=["Triton (coord-fused)", "Torch (grid_sample)"],
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="Latency (ms)",
                    plot_name=f"coord-attn-B{BATCH}-H{N_HEADS}-D{HEAD_DIM}-{mode}-W{W_IMG}-P{P_samples}",
                    args={
                        "BATCH": BATCH,
                        "N_HEADS": N_HEADS,
                        "HEAD_DIM": HEAD_DIM,
                        "W_IMG": W_IMG,
                        "H_IMG": W_IMG,
                        "P_samples": P_samples,
                        "mode": mode,
                    },
                )
            )

    @testing.perf_report(configs)
    def bench_coord_attention(BATCH, N_HEADS, W_IMG, H_IMG, HEAD_DIM, S_views, P_samples, mode, provider, device="cuda"):
        torch.cuda.synchronize()

        q, k, v, coords = gen_inputs(BATCH, S_views, N_HEADS, W_IMG, H_IMG, HEAD_DIM, P_samples, device, DTYPE)

        if S_views == max(S_LIST):
            chk = check_correctness(q, k, v, coords, W_IMG, H_IMG, SM_SCALE)
            print(f"[Check] B={BATCH}, heads={N_HEADS}, S={S_views}, W={W_IMG}, H={H_IMG}, P={P_samples}, D={HEAD_DIM} | "
                f"out:{chk['out_ok']}({chk['out_max']:.2e}) "
                f"dq:{chk['dq_ok']}({chk['dq_max']:.2e}) "
                f"dk:{chk['dk_ok']}({chk['dk_max']:.2e}) "
                f"dv:{chk['dv_ok']}({chk['dv_max']:.2e})")

        if provider == "triton-coord":
            if mode == "fwd":
                fn = lambda: fused_deform_sdpa(q, k, v, coords, W_IMG, H_IMG, SM_SCALE)
            else:
                o = fused_deform_sdpa(q, k, v, coords, W_IMG, H_IMG, SM_SCALE)
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        elif provider == "torch-ref":
            if mode == "fwd":
                fn = lambda: ref_deform_sdpa(q, k, v, coords, W_IMG, H_IMG, SM_SCALE)
            else:
                o = ref_deform_sdpa(q, k, v, coords, W_IMG, H_IMG, SM_SCALE)
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
        else:
            raise ValueError(provider)

        ms = testing.do_bench(fn, warmup=50, rep=200)

        torch.cuda.reset_peak_memory_stats(device)
        _ = fn()
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"[Mem] provider={provider:12s} mode={mode:3s} | "
            f"B={BATCH} heads={N_HEADS} S={S_views} W={W_IMG} H={H_IMG} P={P_samples} D={HEAD_DIM} | "
            f"peak={peak_mb:.1f} MB", file=open(os.path.join(SAVE_PATH, 'logs.txt'), "a"))

        return ms
    
    os.makedirs(SAVE_PATH, exist_ok=True)
    for filename in os.listdir(SAVE_PATH):
        file_path = os.path.join(SAVE_PATH, filename)
        os.remove(file_path)
    bench_coord_attention.run(save_path=SAVE_PATH, print_data=True)
    
def gen_inputs(batch, S_views, N_HEADS, W_IMG, H_IMG, HEAD_DIM, P_samples, device, dtype=torch.float32):
    L = W_IMG * H_IMG
    q = torch.randn(batch, S_views, N_HEADS, L, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    coords = torch.rand(batch, S_views, L, S_views, P_samples, 2, device=device, dtype=dtype)
    return q, k, v, coords

def check_correctness(q, k, v, coords, W_IMG, H_IMG, sm_scale, atol=1e-2, rtol=1e-2):
    dout = torch.randn_like(q)
    out_ref = ref_deform_sdpa(q, k, v, coords, W_IMG, H_IMG, sm_scale)
    out_tri = fused_deform_sdpa(q, k, v, coords, W_IMG, H_IMG, sm_scale)
    out_ok = torch.allclose(out_ref, out_tri, atol=atol, rtol=rtol)
    out_max = (out_ref - out_tri).abs().max().item()

    q_ref, k_ref, v_ref = (q.detach().clone().requires_grad_(True),
                           k.detach().clone().requires_grad_(True),
                           v.detach().clone().requires_grad_(True))
    out_ref2 = ref_deform_sdpa(q_ref, k_ref, v_ref, coords.detach().clone(), W_IMG, H_IMG, sm_scale)
    out_ref2.backward(dout)
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    q_tri, k_tri, v_tri = (q.detach().clone().requires_grad_(True),
                           k.detach().clone().requires_grad_(True),
                           v.detach().clone().requires_grad_(True))
    out_tri2 = fused_deform_sdpa(q_tri, k_tri, v_tri, coords.detach().clone(), W_IMG, H_IMG, sm_scale)
    out_tri2.backward(dout)
    dq_tri, dk_tri, dv_tri = q_tri.grad, k_tri.grad, v_tri.grad

    dq_ok = torch.allclose(dq_ref, dq_tri, atol=atol, rtol=rtol)
    dk_ok = torch.allclose(dk_ref, dk_tri, atol=atol, rtol=rtol)
    dv_ok = torch.allclose(dv_ref, dv_tri, atol=atol, rtol=rtol)
    dq_max = (dq_ref - dq_tri).abs().max().item()
    dk_max = (dk_ref - dk_tri).abs().max().item()
    dv_max = (dv_ref - dv_tri).abs().max().item()

    return {
        "out_ok": out_ok, "out_max": out_max,
        "dq_ok": dq_ok, "dq_max": dq_max,
        "dk_ok": dk_ok, "dk_max": dk_max,
        "dv_ok": dv_ok, "dv_max": dv_max,
    }