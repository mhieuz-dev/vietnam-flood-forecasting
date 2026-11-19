import json
from datetime import date

import numpy as np
import torch

from floodlab import dataset, du_doan, huan_luyen, mau

P = 224


def _mau_tho():
    rng = np.random.default_rng(0)
    return {"tinh": rng.normal(5, 2, (len(mau.KENH_TINH), P, P)).astype(np.float32),
            "s1": np.full((2, P, P), -15.0, np.float32), "nhan": np.zeros((P, P), np.uint8),
            "nhan_trang_thai": np.zeros((P, P), np.uint8), "quan_trac": rng.normal(10, 3, (31, 10)).astype(np.float32),
            "du_bao": np.full((3, 2), 10.0, np.float32), "mat_na": np.array([True, True, False]),
            "chi_muc": {"tile_id": "huong_1_1", "ngay_t": date(2020, 10, 13), "L": 2, "khoang_cach_s1_ngay": 6}}


def _ckpt(tmp_path, monkeypatch, T):
    monkeypatch.setattr(du_doan, "DIR_CKPT", tmp_path)
    f = tmp_path / "chuan_hoa.json"
    f.write_text(json.dumps({"tinh_tb": [5.0] * 11, "tinh_ls": [2.0] * 11, "qt_tb": [10.0] * 10, "qt_ls": [3.0] * 10}))
    monkeypatch.setattr(dataset, "F_CHUAN_HOA", f)
    cfg = {"ten": "thu", "mo_hinh": "unet_b3", "bo": ["cn"]}
    torch.manual_seed(0)
    m = huan_luyen.tao_mo_hinh(cfg)
    (tmp_path / "thu").mkdir()
    torch.save({"mo_hinh": m.state_dict(), "cfg": cfg, "nhiet_do": T}, tmp_path / "thu" / "best.pt")
    return m


def test_du_doan_tra_xac_suat_da_hieu_chinh_nhiet_do(tmp_path, monkeypatch):
    m = _ckpt(tmp_path, monkeypatch, 2.0)
    f = du_doan.ham_du_doan("thu", thiet_bi="cpu")
    p = f(_mau_tho())
    assert p.shape == (P, P) and p.dtype == np.float32 and (p >= 0).all() and (p <= 1).all()
    from types import SimpleNamespace
    b = dataset.mau_cho_mo_hinh(_mau_tho(), SimpleNamespace(**_mau_tho()["chi_muc"]), bo={"cn"})
    with torch.no_grad():
        z = m.eval()(b)[0, 0]
    assert np.allclose(p, torch.sigmoid(z / 2.0).numpy(), atol=1e-5)  # đúng bo của cfg và chia nhiệt độ


def test_du_doan_bien_doi_dau_vao_truoc_khi_vao_mo_hinh(tmp_path, monkeypatch):
    # phép thử đất (nhiễu, hoán vị) cắm vào đây: bien_doi nhận lô đã chuẩn hóa và chỉ mục của mẫu
    _ckpt(tmp_path, monkeypatch, 1.0)
    goi = []
    f = du_doan.ham_du_doan("thu", thiet_bi="cpu", bien_doi=lambda b, r: goi.append(r.tile_id) or b)
    f(_mau_tho())
    assert goi == ["huong_1_1"]
