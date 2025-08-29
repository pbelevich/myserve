import torch, pytest
from torch import nn
from myserve.core.quant.qlinear import W8Linear, W4Linear

@pytest.mark.parametrize("cls", [W8Linear, W4Linear])
@torch.no_grad()
def test_qlinear_close_to_fp(cls):
    lin = nn.Linear(64, 32, bias=True)
    qlin = cls(lin, act_dtype=torch.float16)
    x = torch.randn(4, 64, dtype=torch.float16)
    y_fp = lin(x.to(torch.float32)) # baseline in fp32 for stability
    y_q = qlin(x)
    # relative error tolerance is loose; tiny layers + per‑row symmetric
    rel = (y_fp - y_q.float()).abs().mean() / y_fp.abs().clamp(min=1e-6).mean()
    assert rel < 0.1
