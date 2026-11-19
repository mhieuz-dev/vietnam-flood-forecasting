import torch

from floodlab.mo_hinh.film import FiLM
from floodlab.mo_hinh.hop_nhat import MoHinhHopNhat
from floodlab.mo_hinh.unet import UNetB3


def _batch(B=2, L=2):
    torch.manual_seed(0)
    mn = torch.tensor([[k < L for k in range(3)]] * B)
    return {"s1": torch.randn(B, 2, 224, 224), "dem": torch.randn(B, 1, 224, 224), "tinh": torch.randn(B, 11, 224, 224),
            "quan_trac": torch.randn(B, 31, 10), "du_bao": torch.randn(B, 3, 2) * mn[..., None], "mat_na_du_bao": mn,
            "L": torch.full((B,), L), "khoang_cach_s1": torch.rand(B)}


def test_hinh_dang_dau_ra():
    b = _batch()
    for m in (MoHinhHopNhat(bo_ma_hoa="terramind_v1_tiny", pretrained=False), UNetB3()):
        m.eval()
        with torch.no_grad():
            assert m(b).shape == (2, 1, 224, 224)


def test_film_khoi_tao_dong_nhat():
    f = FiLM(16, 8)
    x = torch.randn(3, 8, 5, 5)
    assert torch.allclose(f(x, torch.randn(3, 16)), x)


def test_bo_dat_lam_dau_vao_bang_0():
    m = MoHinhHopNhat(bo_ma_hoa="terramind_v1_tiny", pretrained=False, bo={"dat"})
    b = _batch()
    tinh = m.chuan_bi_tinh(b["tinh"])
    assert torch.all(tinh[:, m.KENH_DAT] == 0) and torch.any(tinh[:, 0] != 0)


def test_mat_na_du_bao_dung_L():
    m = MoHinhHopNhat(bo_ma_hoa="terramind_v1_tiny", pretrained=False).eval()
    b = _batch(L=1)
    with torch.no_grad():
        y1 = m(b)
        b2 = dict(b)
        b2["du_bao"] = b["du_bao"].clone()
        b2["du_bao"][:, 1:] = torch.randn(2, 2, 2) * 100
        y2 = m(b2)
    assert torch.allclose(y1, y2, atol=1e-5)


def _dau_vao_film0(m, b):
    """max|đặc trưng lưng| đi vào FiLM đầu tiên của bộ giải mã (chế độ train, thống kê theo batch)."""
    vao = []
    h = m.giai_ma.film0.register_forward_hook(lambda _, a, __: vao.append(float(a[0].abs().max())))
    m.train()
    with torch.no_grad():
        m(b)
    h.remove()
    return vao[0]


def test_dac_trung_lung_duoc_chuan_hoa_khi_trong_so_gop_phinh():
    # hop_nhat_gfs_s0 (30/10): conv gộp không có chuẩn hóa nên đầu ra phình 5,7 -> 1325 sau 8654 bước, conv đầu bộ
    # giải mã vượt 65504 của fp16 và ra NaN. Trọng số conv gộp phình 1000 lần không được đổi thang đặc trưng.
    from floodlab.mo_hinh.prithvi_b4 import MoHinhB4

    b = _batch(B=4)
    b["s2"] = torch.randn(4, 6, 224, 224)
    for m in (MoHinhHopNhat(bo_ma_hoa="terramind_v1_tiny", pretrained=False), MoHinhB4(pretrained=False, nho_de_test=True)):
        with torch.no_grad():
            for c in m.gop_lung.modules():
                if isinstance(c, torch.nn.Conv2d):
                    c.weight.mul_(1000)
        assert _dau_vao_film0(m, b) < 20
