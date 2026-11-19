"""B5: hai nhánh U-TAE theo cấu hình two_utae của GFF (Victor và cs., NeurIPS 2024; gff/models/two_backbones.py,
configs/two_utae_*.yml), mã U-TAE chép nguyên văn ở floodlab/mo_hinh/gff.

Chuyển đổi đầu vào (khác GFF, ghi trong bài): khí tượng của bài này là chuỗi theo ô (31 ngày quan trắc + 3 ngày dự báo,
12 đặc trưng/bước như MaHoaKhiTuong.chuoi) chứ không phải lưới ERA5 thô, nên nhánh ngữ cảnh nhận chuỗi đó trải đều
trên lưới 8 x 8 (GFF có tùy chọn average_context tương đương). Nhánh cục bộ nhận S1, DEM và 11 lớp tĩnh (cùng đầu vào
với mô hình chính) cộng 128 kênh ngữ cảnh phóng lên 224 x 224. Điều kiện thời hạn: nhúng số ngày từ ảnh S1 trạng thái
tới ngày t (s1_lead_days của GFF), bảng 35 dòng, quá hạn gộp vào dòng cuối. bo={"khi_tuong"}: bỏ hẳn nhánh ngữ cảnh
như các cấu hình no_ERA5 của GFF.
"""
import torch
import torch.nn.functional as F
from torch import nn

from floodlab.mo_hinh.chung import MaHoaKhiTuong
from floodlab.mo_hinh.gff.utae import UTAE
from floodlab.mo_hinh.hop_nhat import MoHinhHopNhat

LUOI_NGU_CANH, D_NGU_CANH, D_THOI_HAN, SO_BUOC = 8, 128, 16, 34


class MoHinhB5(nn.Module):
    KENH_DAT, KENH_CN, KENH_DIA_HINH = MoHinhHopNhat.KENH_DAT, MoHinhHopNhat.KENH_CN, MoHinhHopNhat.KENH_DIA_HINH

    def __init__(self, bo: set[str] = frozenset()):
        super().__init__()
        self.bo = set(bo)
        self.co_ngu_canh = "khi_tuong" not in self.bo
        self.so_thoi_han = SO_BUOC + 1
        self.nhung_thoi_han = nn.Embedding(self.so_thoi_han, D_THOI_HAN)
        if self.co_ngu_canh:
            self.ngu_canh = UTAE(12, encoder_widths=[32, 32], decoder_widths=[32, 32], out_conv=[D_NGU_CANH],
                                 cond_dim=D_THOI_HAN, temp_encoding="ltae", cond_norm_affine=None)
        self.cuc_bo = UTAE(14 + D_NGU_CANH * self.co_ngu_canh, encoder_widths=[64, 64, 64, 128],
                           decoder_widths=[64, 64, 64, 128], out_conv=[64, 1], cond_dim=D_THOI_HAN,
                           temp_encoding="ltae", cond_norm_affine=None)

    def forward(self, b: dict) -> torch.Tensor:
        s1 = torch.nan_to_num(b["s1"].float())
        dem = torch.nan_to_num(b["dem"].float())
        if "s1" in self.bo:
            s1 = torch.zeros_like(s1)
        if "dia_hinh" in self.bo:
            dem = torch.zeros_like(dem)
        tinh = MoHinhHopNhat.chuan_bi_tinh(self, b["tinh"])
        B, _, H, W = s1.shape
        ngay = (b["khoang_cach_s1"].float() * 30).round().long().clamp(0, self.so_thoi_han - 1)
        dk = self.nhung_thoi_han(ngay)
        cuc_bo = [s1, dem, tinh]
        if self.co_ngu_canh:
            chuoi = MaHoaKhiTuong.chuoi(b)  # (B, 34, 12)
            vao = chuoi[..., None, None].expand(-1, -1, -1, LUOI_NGU_CANH, LUOI_NGU_CANH)
            vt = torch.arange(chuoi.shape[1], device=s1.device).repeat(B, 1)
            nc = self.ngu_canh(vao, batch_positions=vt, lead=dk)[:, 0]
            cuc_bo.insert(0, F.interpolate(nc, size=(H, W), mode="nearest"))
        x = torch.cat(cuc_bo, 1)[:, None]  # một bước thời gian, như GFF
        vt1 = torch.zeros(B, 1, dtype=torch.long, device=s1.device)
        return self.cuc_bo(x, batch_positions=vt1, lead=dk)[:, 0]
