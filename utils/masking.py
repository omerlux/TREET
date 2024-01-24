import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu", diagonal=1):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=diagonal).to(device)

    @property
    def mask(self):
        return self._mask


class DoubleTriangularCausalMask():
    def __init__(self, B, L, device="cpu", diagonal=1, history=0):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            mask_causal = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=diagonal)
            mask_lower = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=history + 1).permute(0, 1, 3, 2)
            self._mask = torch.bitwise_xor(mask_causal, mask_lower).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
