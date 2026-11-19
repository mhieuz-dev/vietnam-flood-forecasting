"""B4: Prithvi-EO-2.0-300M (chỉ nhận 6 băng quang học HLS) với ảnh S2 mùa khô + cùng nhánh tĩnh, khí tượng, FiLM."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from floodlab.mo_hinh.chung import GiaiMa, MaHoaKhiTuong
from floodlab.mo_hinh.hop_nhat import MoHinhHopNhat, _khoi_conv


class MoHinhB4(nn.Module):
    TANG_LAY = (5, 11, 17, 23)

    def __init__(self, pretrained: bool = True, nho_de_test: bool = False, bo: set[str] = frozenset(),
                 ckpt_lung: bool = False):
        super().__init__()
        from terratorch.registry import BACKBONE_REGISTRY

        self.bo = set(bo)
        ten = "terratorch_prithvi_eo_tiny" if nho_de_test else "terratorch_prithvi_eo_v2_300"
        self.lung = BACKBONE_REGISTRY.build(ten, pretrained=pretrained and not nho_de_test, num_frames=1)
        with torch.no_grad():
            out = self.lung(torch.zeros(1, 6, 224, 224))
        self.tang = self.TANG_LAY if len(out) > max(self.TANG_LAY) else tuple(range(len(out)))[-4:]
        if ckpt_lung:  # batch 16 tràn 8 GB laptop (04/11): tính lại activation từng khối ViT khi lan truyền ngược
            for k in self.lung.blocks:  # gán forward trên thể hiện để khóa state_dict giữ nguyên
                k.forward = (lambda f: lambda *a, **kw: checkpoint(f, *a, use_reentrant=False, **kw)
                             if torch.is_grad_enabled() else f(*a, **kw))(k.forward)
        d = out[-1].shape[-1]
        # BatchNorm sau conv gộp: thiếu nó thì đặc trưng lưng phình không chặn và tràn fp16 (NaN epoch 6, 30/10)
        self.gop_lung = nn.Sequential(nn.Conv2d(d * len(self.tang), 256, 1), nn.BatchNorm2d(256))
        self.tinh = nn.ModuleList([_khoi_conv(12, 32, 1), _khoi_conv(32, 64, 2), _khoi_conv(64, 96, 2), _khoi_conv(96, 128, 2)])
        self.khi_tuong = MaHoaKhiTuong(dung_khi_tuong="khi_tuong" not in self.bo)
        self.giai_ma = GiaiMa(256, [128, 96, 64, 32], [192, 128, 64, 32])

    def forward(self, b: dict) -> torch.Tensor:
        s2 = torch.nan_to_num(b["s2"].float())
        tok = self.lung(s2)
        B = s2.shape[0]
        lay = []
        for i in self.tang:
            t = tok[i][:, -196:]  # bỏ token CLS
            lay.append(t.transpose(1, 2).reshape(B, -1, 14, 14))
        day = self.gop_lung(torch.cat(lay, 1))
        x = torch.cat([torch.nan_to_num(b["dem"].float()), MoHinhHopNhat.chuan_bi_tinh(self, b["tinh"])], 1)
        skip = []
        for k in self.tinh:
            x = k(x)
            skip.append(x)
        return self.giai_ma(day, skip[::-1], self.khi_tuong(b))


MoHinhB4.KENH_DAT, MoHinhB4.KENH_CN, MoHinhB4.KENH_DIA_HINH = MoHinhHopNhat.KENH_DAT, MoHinhHopNhat.KENH_CN, MoHinhHopNhat.KENH_DIA_HINH
