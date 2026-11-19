"""B3: U-Net với bộ mã hóa ResNet-34 huấn luyện từ đầu (không trọng số ImageNet), cùng nhánh khí tượng và FiLM."""
import torch
from torch import nn
from torchvision.models import resnet34

from floodlab.mo_hinh.chung import GiaiMa, MaHoaKhiTuong
from floodlab.mo_hinh.hop_nhat import MoHinhHopNhat


class UNetB3(nn.Module):
    def __init__(self, c_vao: int = 14, bo: set[str] = frozenset()):
        super().__init__()
        self.bo = set(bo)
        r = resnet34(weights=None)
        r.conv1 = nn.Conv2d(c_vao, 64, 7, stride=2, padding=3, bias=False)
        self.vao = nn.Sequential(r.conv1, r.bn1, r.relu)        # 112, 64
        self.l1 = nn.Sequential(r.maxpool, r.layer1)          # 56, 64
        self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4  # 28/128, 14/256, 7/512
        self.goc = nn.Sequential(nn.Conv2d(c_vao, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU())  # 224, 32
        self.khi_tuong = MaHoaKhiTuong(dung_khi_tuong="khi_tuong" not in self.bo)
        self.giai_ma = GiaiMa(512, [256, 128, 64, 64, 32], [256, 128, 64, 64, 32])

    def forward(self, b: dict) -> torch.Tensor:
        tinh = MoHinhHopNhat.chuan_bi_tinh(self, b["tinh"])
        s1 = torch.nan_to_num(b["s1"].float())
        dem = torch.nan_to_num(b["dem"].float())
        if "s1" in self.bo:
            s1 = torch.zeros_like(s1)
        if "dia_hinh" in self.bo:
            dem = torch.zeros_like(dem)
        x = torch.cat([s1, dem, tinh], 1)
        s224 = self.goc(x)
        s112 = self.vao(x)
        s56 = self.l1(s112)
        s28 = self.l2(s56)
        s14 = self.l3(s28)
        s7 = self.l4(s14)
        return self.giai_ma(s7, [s14, s28, s56, s112, s224], self.khi_tuong(b))


UNetB3.KENH_DAT, UNetB3.KENH_CN, UNetB3.KENH_DIA_HINH = MoHinhHopNhat.KENH_DAT, MoHinhHopNhat.KENH_CN, MoHinhHopNhat.KENH_DIA_HINH
