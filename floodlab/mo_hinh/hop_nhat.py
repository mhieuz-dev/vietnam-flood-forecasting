"""Mô hình chính: TerraMind (S1GRD + DEM) + nhánh tĩnh CNN + nhánh khí tượng, hợp nhất bằng FiLM ở bộ giải mã."""
import torch
from torch import nn

from floodlab.mo_hinh.chung import D_DK, GiaiMa, MaHoaKhiTuong

# Thứ tự 11 kênh 'tinh' (12 lớp tĩnh trừ DEM): do_doc, hand, log_dien_tich_thuong_nguon, set_0_30, cat_0_30,
# dung_trong_0_30, set_30_100, cat_30_100, cn, worldcover, nuoc_thuong_xuyen
KENH_TINH = ("do_doc", "hand", "log_dien_tich_thuong_nguon", "set_0_30", "cat_0_30", "dung_trong_0_30",
             "set_30_100", "cat_30_100", "cn", "worldcover", "nuoc_thuong_xuyen")


def _khoi_conv(c_vao, c_ra, buoc):
    return nn.Sequential(nn.Conv2d(c_vao, c_ra, 3, stride=buoc, padding=1), nn.BatchNorm2d(c_ra), nn.GELU(),
                         nn.Conv2d(c_ra, c_ra, 3, padding=1), nn.BatchNorm2d(c_ra), nn.GELU())


class MoHinhHopNhat(nn.Module):
    KENH_DAT = [3, 4, 5, 6, 7, 8]       # đất thô + CN
    KENH_CN = [8]
    KENH_DIA_HINH = [0, 1, 2]           # cùng với DEM
    TANG_LAY = (2, 5, 8, 11)

    def __init__(self, bo_ma_hoa: str = "terramind_v1_base", pretrained: bool = True, bo: set[str] = frozenset()):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.bo = set(bo)
        self.lung = BACKBONE_REGISTRY.build(f"terratorch_{bo_ma_hoa}", pretrained=pretrained, modalities=["S1GRD", "DEM"])
        with torch.no_grad():
            d = self.lung({"S1GRD": torch.zeros(1, 2, 224, 224), "DEM": torch.zeros(1, 1, 224, 224)})[-1].shape[-1]
        # BatchNorm sau conv gộp: thiếu nó thì đặc trưng lưng phình không chặn và tràn fp16 (NaN epoch 6, 30/10)
        self.gop_lung = nn.Sequential(nn.Conv2d(d * len(self.TANG_LAY), 256, 1), nn.BatchNorm2d(256))
        self.tinh = nn.ModuleList([_khoi_conv(11, 32, 1), _khoi_conv(32, 64, 2), _khoi_conv(64, 96, 2), _khoi_conv(96, 128, 2)])
        self.khi_tuong = MaHoaKhiTuong(dung_khi_tuong="khi_tuong" not in self.bo)
        self.giai_ma = GiaiMa(256, [128, 96, 64, 32], [192, 128, 64, 32])

    def chuan_bi_tinh(self, tinh: torch.Tensor) -> torch.Tensor:
        tinh = torch.nan_to_num(tinh.float()).clone()
        if "dat" in self.bo:
            tinh[:, self.KENH_DAT] = 0
        if "cn" in self.bo:
            tinh[:, self.KENH_CN] = 0
        if "dia_hinh" in self.bo:
            tinh[:, self.KENH_DIA_HINH] = 0
        return tinh

    def forward(self, b: dict) -> torch.Tensor:
        s1 = torch.nan_to_num(b["s1"].float())
        dem = torch.nan_to_num(b["dem"].float())
        if "s1" in self.bo:
            s1 = torch.zeros_like(s1)
        if "dia_hinh" in self.bo:
            dem = torch.zeros_like(dem)
        tok = self.lung({"S1GRD": s1, "DEM": dem})
        B = s1.shape[0]
        anh = torch.cat([tok[i].transpose(1, 2).reshape(B, -1, 14, 14) for i in self.TANG_LAY], 1)
        day = self.gop_lung(anh)
        x = self.chuan_bi_tinh(b["tinh"])
        skip = []
        for k in self.tinh:
            x = k(x)
            skip.append(x)  # 224, 112, 56, 28
        c = self.khi_tuong(b)
        return self.giai_ma(day, skip[::-1], c)
