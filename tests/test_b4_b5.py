import torch

from floodlab import s2
from floodlab.mo_hinh.prithvi_b4 import MoHinhB4


def test_b4_hinh_dang():
    m = MoHinhB4(pretrained=False, nho_de_test=True).eval()
    B = 2
    b = {"s2": torch.randn(B, 6, 224, 224), "tinh": torch.randn(B, 11, 224, 224), "dem": torch.randn(B, 1, 224, 224),
         "quan_trac": torch.randn(B, 31, 10), "du_bao": torch.randn(B, 3, 2), "mat_na_du_bao": torch.ones(B, 3, dtype=torch.bool),
         "L": torch.full((B,), 3), "khoang_cach_s1": torch.rand(B)}
    with torch.no_grad():
        assert m(b).shape == (B, 1, 224, 224)


def test_s2_chi_lay_mua_kho():
    tu, den = s2.khoang_mua_kho(2020)
    assert tu.isoformat() == "2020-01-01" and den.isoformat() == "2020-04-30"
    assert s2.BANG_HLS == ("B2", "B3", "B4", "B8A", "B11", "B12")



def test_s2_mat_na_may_chi_giu_thuc_vat_dat_nuoc():
    # Lớp 7 (chưa phân loại) và 11 (tuyết) giữ lại mây sót: mảng trắng ở Trà Khúc 2018 (03/11); VN không có tuyết
    assert s2.LOP_SCL_SACH == (4, 5, 6) and s2.SO_ANH_TOI_THIEU == 3 and s2.NGUONG_XS_MAY == 40


def test_s2_trong_thanh_nan_bu_nam_truoc_va_luu_float16(monkeypatch, tmp_path):
    # computePixels trả 0 cho pixel bị che: 0 ở cả 6 băng phải thành NaN; ty_le_bu ghi phần pixel lấy từ mùa khô
    # năm n-1 (Trà Khúc 2018 trống 80%); lưu float16 (14.808 cặp ô-năm x 1,2 MB float32 = 17,8 GB)
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setattr(s2, "THU_MUC", tmp_path)
    monkeypatch.setattr(s2, "_anh", lambda o, nam: None)
    raw = np.zeros((224, 224), dtype=[(b, "f8") for b in (*s2.BANG_HLS, "bu")])
    for i, b in enumerate(s2.BANG_HLS):
        raw[b] = 1000.0 * (i + 1)
        raw[b][0, :] = 0  # hàng đầu không có ảnh ở cả hai năm
    raw["bu"][1:57, :] = 1  # một phần tư ô lấy từ năm trước

    def gia_tai(anh, goc, rong, cao, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, raw)
        return p
    monkeypatch.setattr(s2, "tai_luoi", gia_tai)
    o = SimpleNamespace(bounds=(0.0, 0.0, 4480.0, 4480.0))
    a = s2.anh_mua_kho("t_1", o, 2020)
    assert a.dtype == np.float32 and a.shape == (6, 224, 224)
    assert np.isnan(a[:, 0]).all() and not np.isnan(a[:, 1:]).any()
    assert abs(s2.ty_le_bu("t_1", 2020) - 56 / 224) < 1e-6
    with np.load(tmp_path / "t_1" / "2020.npz") as z:
        assert z["anh"].dtype == np.float16
    b = s2.anh_mua_kho("t_1", o, 2020)  # đọc lại từ đĩa
    assert b.dtype == np.float32 and abs(float(b[5, 5, 5]) - 0.6) < 1e-3 and np.isnan(b[:, 0]).all()


def test_b4_checkpoint_lung_cung_ket_qua_va_gradient():
    # Prithvi-300M batch 16 tràn 8 GB laptop (OOM 04/11): checkpoint từng khối ViT giữ nguyên batch và công thức,
    # chỉ đổi bộ nhớ lấy thời gian; đầu ra và gradient phải trùng bản không checkpoint
    torch.manual_seed(0)
    a = MoHinhB4(pretrained=False, nho_de_test=True).train()
    b = MoHinhB4(pretrained=False, nho_de_test=True, ckpt_lung=True).train()
    b.load_state_dict(a.state_dict())
    B = 2
    x = {"s2": torch.randn(B, 6, 224, 224), "tinh": torch.randn(B, 11, 224, 224), "dem": torch.randn(B, 1, 224, 224),
         "quan_trac": torch.randn(B, 31, 10), "du_bao": torch.randn(B, 3, 2), "mat_na_du_bao": torch.ones(B, 3, dtype=torch.bool),
         "L": torch.full((B,), 3), "khoang_cach_s1": torch.rand(B)}
    ya, yb = a(x), b(x)
    assert torch.allclose(ya, yb, atol=1e-5)
    ya.sum().backward()
    yb.sum().backward()
    ga, gb = a.lung.blocks[0].attn.qkv.weight.grad, b.lung.blocks[0].attn.qkv.weight.grad
    assert gb is not None and torch.allclose(ga, gb, atol=1e-4)


def _lo_b5(B=2):
    return {"s1": torch.randn(B, 2, 224, 224), "tinh": torch.randn(B, 11, 224, 224), "dem": torch.randn(B, 1, 224, 224),
            "quan_trac": torch.randn(B, 31, 10), "du_bao": torch.randn(B, 3, 2),
            "mat_na_du_bao": torch.tensor([[True, False, False], [True, True, True]][:B]),
            "L": torch.tensor([1, 3][:B]), "khoang_cach_s1": torch.tensor([0.2, 0.5][:B])}


def test_b5_hinh_dang_va_gradient():
    # B5: hai nhánh U-TAE của GFF (ngữ cảnh: chuỗi khí tượng theo thời gian; cục bộ: S1 + DEM + lớp tĩnh), điều kiện
    # theo khoảng cách ảnh S1 tới ngày t như s1_lead_days của GFF; đầu ra logit (B, 1, 224, 224) như các mô hình khác
    from floodlab.mo_hinh.utae_b5 import MoHinhB5
    m = MoHinhB5().train()
    y = m(_lo_b5())
    assert y.shape == (2, 1, 224, 224) and torch.isfinite(y).all()
    y.mean().backward()
    assert all(p.grad is not None for p in m.ngu_canh.parameters() if p.requires_grad)


def test_b5_bo_khi_tuong_khong_doi_theo_chuoi():
    from floodlab.mo_hinh.utae_b5 import MoHinhB5
    m = MoHinhB5(bo={"khi_tuong"}).eval()
    b = _lo_b5()
    with torch.no_grad():
        y1 = m(b)
        b["quan_trac"] = b["quan_trac"] * 5 + 3
        b["du_bao"] = b["du_bao"] * 5
        y2 = m(b)
    assert torch.allclose(y1, y2)


def test_s2_vo_cuc_thanh_nan_ca_file_cu(monkeypatch, tmp_path):
    # Sau toFloat, computePixels trả -inf (không phải 0) cho pixel bị che ở cả hai năm: 11.481/14.808 file có -inf,
    # nan_to_num đổi thành -3,4e38 làm Prithvi ra NaN cả 3 lượt B4 (05/11). Sửa khi tải mới VÀ khi đọc file đã lưu.
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setattr(s2, "THU_MUC", tmp_path)
    monkeypatch.setattr(s2, "_anh", lambda o, nam: None)
    raw = np.zeros((224, 224), dtype=[(b, "f8") for b in (*s2.BANG_HLS, "bu")])
    for i, b in enumerate(s2.BANG_HLS):
        raw[b] = 1000.0 * (i + 1)
        raw[b][:3] = -np.inf

    def gia_tai(anh, goc, rong, cao, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, raw)
        return p
    monkeypatch.setattr(s2, "tai_luoi", gia_tai)
    a = s2.anh_mua_kho("t_1", SimpleNamespace(bounds=(0.0, 0.0, 4480.0, 4480.0)), 2020)
    assert np.isnan(a[:, :3]).all() and np.isfinite(a[:, 3:]).all()
    cu = np.full((6, 224, 224), 0.1, np.float16)
    cu[:, 0, 0] = -np.inf
    (tmp_path / "t_2").mkdir()
    np.savez(tmp_path / "t_2" / "2020.npz", anh=cu, ty_le_bu=np.float32(0))
    b = s2.anh_mua_kho("t_2", None, 2020)
    assert np.isnan(b[:, 0, 0]).all() and np.isfinite(b[:, 1:]).all()
