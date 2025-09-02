import torch
from einops import rearrange, einsum
import torch.nn.functional as F

def deform_sdpa_torch(q, k, v, coords, WIDTH, HEIGHT, sm_scale):
    Z, S, H, L, D = q.shape
    P = coords.shape[-2]
    k_map = rearrange(k, 'z s head (h w) d -> (z head) s d h w', h=HEIGHT, w=WIDTH).to(torch.float32)
    v_map = rearrange(v, 'z s head (h w) d -> (z head) s d h w', h=HEIGHT, w=WIDTH).to(torch.float32)
    grid = coords.unsqueeze(2).expand(Z, S, H, L, S, P, 2)
    grid = rearrange(grid, 'z s1 head L s2 p d -> (z head) (s1 L) s2 p d').to(torch.float32)
    k_sampled, v_sampled = [], []
    for si in range(S):
        k_sampled.append(F.grid_sample(k_map[:,si], grid[:,:,si] * 2 - 1, align_corners=True))
        v_sampled.append(F.grid_sample(v_map[:,si], grid[:,:,si] * 2 - 1, align_corners=True))
    k_sampled = torch.cat(k_sampled, dim=-1)
    v_sampled = torch.cat(v_sampled, dim=-1)
    k_ = rearrange(k_sampled, '(z head) d sl sp -> z head sl sp d', head=H)
    v_ = rearrange(v_sampled, '(z head) d sl sp -> z head sl sp d', head=H)
    q_ = rearrange(q.to(torch.float32), 'z s head l d -> z head (s l) d', head=H) * sm_scale * 1.4426950408889634
    p = einsum(q_, k_, '... d, ... n d -> ... n')
    m = p.amax(dim=-1, keepdim=True)
    p_unn = torch.pow(2.0, (p - m) )
    sum_p = p_unn.sum(dim=-1, keepdim=True)
    p_norm = (p_unn / sum_p)
    out_ref = einsum(p_norm, v_, '... n, ... n d -> ... d')
    out_ref = rearrange(out_ref, 'z head (s l) d -> z s head l d', s=S)
    return out_ref.to(q.dtype)
