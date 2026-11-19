"""Khối dùng chung: bộ mã hóa khí tượng (GRU) và bộ giải mã kiểu U-Net có FiLM ở mọi tầng."""
import torch
from torch import nn
from torch.nn import functional as F

from floodlab.mo_hinh.film import FiLM

D_DK = 128


class MaHoaKhiTuong(nn.Module):
    """Chuỗi 31 ngày quan trắc + 3 ngày dự báo (12 đặc trưng/bước, có cờ dự báo và cờ còn hiệu lực)
    -> GRU 2 lớp; nối thời hạn L (one-hot) và khoảng cách ảnh S1 -> vector điều kiện D_DK."""

    def __init__(self, dung_khi_tuong: bool = True):
        super().__init__()
        self.dung = dung_khi_tuong
        self.gru = nn.GRU(12, D_DK, num_layers=2, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D_DK + 4, D_DK), nn.GELU(), nn.Linear(D_DK, D_DK))

    @staticmethod
    def chuoi(b: dict) -> torch.Tensor:
        qt = torch.nan_to_num(b["quan_trac"][..., :9].float())
        B = qt.shape[0]
        mn = b["mat_na_du_bao"].float()
        db = torch.nan_to_num(b["du_bao"].float()) * mn[..., None]  # ngày ngoài thời hạn luôn là 0
        buoc_qt = torch.cat([qt, torch.zeros(B, 31, 1, device=qt.device), torch.ones(B, 31, 1, device=qt.device),
                             torch.zeros(B, 31, 1, device=qt.device)], -1)
        buoc_db = torch.cat([db, torch.zeros(B, 3, 7, device=qt.device), mn[..., None],
                             torch.ones(B, 3, 1, device=qt.device), mn[..., None]], -1)
        return torch.cat([buoc_qt, buoc_db], 1)  # (B, 34, 12)

    def forward(self, b: dict) -> torch.Tensor:
        x = self.chuoi(b)
        if not self.dung:
            x = torch.zeros_like(x)
        _, h = self.gru(x)
        L = F.one_hot((b["L"].long() - 1).clamp(0, 2), 3).float()
        return self.mlp(torch.cat([h[-1], L, b["khoang_cach_s1"].float()[:, None]], -1))


class KhoiLen(nn.Module):
    def __init__(self, c_vao: int, c_bo_qua: int, c_ra: int):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(c_vao + c_bo_qua, c_ra, 3, padding=1), nn.BatchNorm2d(c_ra), nn.GELU(),
                                  nn.Conv2d(c_ra, c_ra, 3, padding=1), nn.BatchNorm2d(c_ra), nn.GELU())
        self.film = FiLM(D_DK, c_ra)

    def forward(self, x, bo_qua, c):
        x = F.interpolate(x, size=bo_qua.shape[-2:], mode="bilinear", align_corners=False)
        return self.film(self.conv(torch.cat([x, bo_qua], 1)), c)


class GiaiMa(nn.Module):
    """Từ đặc trưng thô nhất đi lên qua danh sách skip (thô -> mịn), FiLM ở mỗi tầng, ra logit 1 kênh."""

    def __init__(self, c_day: int, c_bo_qua: list[int], c_tang: list[int]):
        super().__init__()
        self.film0 = FiLM(D_DK, c_day)
        cs = [c_day] + c_tang
        self.khoi = nn.ModuleList(KhoiLen(cs[i], c_bo_qua[i], cs[i + 1]) for i in range(len(c_tang)))
        self.ra = nn.Conv2d(c_tang[-1], 1, 1)

    def forward(self, day, cac_bo_qua, c):
        x = self.film0(day, c)
        for k, s in zip(self.khoi, cac_bo_qua):
            x = k(x, s, c)
        return self.ra(x)
