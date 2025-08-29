import torch
from torch import nn

class W8Linear(nn.Module):
    def __init__(self, linear: nn.Linear, act_dtype: torch.dtype = torch.float16):
        super().__init__()
        W = linear.weight.data # [out,in]
        device = W.device
        out, in_ = W.shape
        # per-out-channel symmetric quant
        max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (max_abs / 127.0).to(act_dtype)
        q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.view(out))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.to(act_dtype))
        else:
            self.bias = None
        self.in_features, self.out_features = in_, out
        self.act_dtype = act_dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[B, in] @ q[out, in]^T -> [B,out]; then scale per-out channel
        x = x.to(self.act_dtype)
        y = torch.matmul(x, self.qweight.t().to(self.act_dtype)) # accumulate in fp32
        y = (y * self.scale) # broadcast over out dim
        if self.bias is not None:
            y = y + self.bias
        return y.to(self.act_dtype)

class W4Linear(nn.Module):
    """Simple 4-bit (no packing) by restricting int8 range to [-8, 7].
    For clarity + portability; you can later add nibble packing if you want.
    """
    def __init__(self, linear: nn.Linear, act_dtype: torch.dtype = torch.float16):
        super().__init__()
        W = linear.weight.data
        out, in_ = W.shape
        max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        # use 7.0 rather than 127.0 to map to [-8,7]
        scale = (max_abs / 7.0).to(act_dtype)
        q = torch.round(W / scale).clamp(-8, 7).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.view(out))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.to(act_dtype))
        else:
            self.bias = None
        self.in_features, self.out_features = in_, out
        self.act_dtype = act_dtype


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.act_dtype)
        y = torch.matmul(x, self.qweight.t().to(self.act_dtype))
        y = (y * self.scale)
        if self.bias is not None:
            y = y + self.bias
        return y.to(self.act_dtype)