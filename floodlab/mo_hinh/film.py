"""FiLM: điều biến đặc trưng ảnh bằng vector điều kiện (khí tượng + thời hạn). Mượn từ GFF (Victor et al., NeurIPS 2024)."""
import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, d_dieu_kien: int, d_dac_trung: int):
        super().__init__()
        self.lin = nn.Linear(d_dieu_kien, 2 * d_dac_trung)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)  # khởi tạo gamma = 1, beta = 0: FiLM là phép đồng nhất

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        dg, b = self.lin(c).chunk(2, dim=-1)
        return x * (1 + dg[..., None, None]) + b[..., None, None]
