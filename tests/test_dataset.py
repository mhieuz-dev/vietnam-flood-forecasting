import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
import torch

from floodlab import dataset, mau

P = 224


def _gia(monkeypatch, tmp_path, n=6):
    rng = np.random.default_rng(0)
    dong = pd.DataFrame({
        "mau_id": [f"m{i}" for i in range(n)], "tile_id": "huong_1_1", "luu_vuc": "huong", "ngay_t": date(2020, 10, 13),
        "L": [1, 2, 3] * (n // 3), "khoang_cach_s1_ngay": 6, "vai_tro": ["train"] * (n - 2) + ["val", "test_b"],
        "ty_le_ngap": [0.2, 0.0, 0.0, 0.0, 0.1, 0.0][:n], "i_o": 0, "i_s1": 0, "i_nhan": 0, "i_mau": range(n)})
    monkeypatch.setattr(dataset, "LUU_VUC", ("huong",))
    monkeypatch.setattr(dataset.mau, "doc_chi_muc", lambda lv: dong)
    monkeypatch.setattr(dataset.mau, "mo_zarr", lambda lv: None)
    nhan = np.zeros((P, P), np.uint8)
    nhan[:20, :10] = 1
    nhan[-5:] = 255

    def doc_mau(g, r, nguon_du_bao="gfs", nguon_quan_trac="imerg_final"):
        tinh = rng.normal(5, 2, (len(mau.KENH_TINH), P, P)).astype(np.float32)
        tinh[0] = 670.665
        mn = np.array([k < r.L for k in range(3)])
        return {"tinh": tinh, "s1": np.stack([np.full((P, P), -12.599), np.full((P, P), -20.293)]).astype(np.float32),
                "nhan": nhan, "nhan_trang_thai": np.zeros((P, P), np.uint8),
                "quan_trac": rng.normal(10, 3, (31, 10)).astype(np.float32),
                "du_bao": (np.full((3, 2), 10.0) * mn[:, None]).astype(np.float32), "mat_na": mn}

    monkeypatch.setattr(dataset.mau, "doc_mau", doc_mau)
    f = tmp_path / "chuan_hoa.json"
    monkeypatch.setattr(dataset, "F_CHUAN_HOA", f)
    f.write_text(json.dumps({"tinh_tb": [5.0] * 11, "tinh_ls": [2.0] * 11, "qt_tb": [10.0] * 10, "qt_ls": [3.0] * 10}))
    return dong


def test_mau_dung_hinh_dang_va_chuan_hoa_terramind(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    ds = dataset.MauNgap("train")
    b = ds[0]
    assert len(ds) == 4 and b["s1"].shape == (2, P, P) and b["dem"].shape == (1, P, P) and b["tinh"].shape == (11, P, P)
    assert b["quan_trac"].shape == (31, 10) and b["du_bao"].shape == (3, 2) and b["mat_na_du_bao"].shape == (3,)
    assert b["nhan"].dtype == torch.long and b["nhan"].shape == (P, P) and b["mau_id"] == "m0"
    assert torch.allclose(b["s1"], torch.zeros_like(b["s1"]), atol=1e-4)  # đúng trung bình tiền huấn luyện -> 0
    assert torch.allclose(b["dem"], torch.zeros_like(b["dem"]), atol=1e-4)


def test_du_bao_ngoai_han_bang_0_sau_chuan_hoa(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    b = dataset.MauNgap("train")[0]  # L = 1
    assert b["L"] == 1 and torch.all(b["du_bao"][1:] == 0) and b["mat_na_du_bao"].tolist() == [True, False, False]


def test_bo_thanh_phan_dien_0(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    b = dataset.MauNgap("train", bo={"dat", "khi_tuong", "s1"})[0]
    dat = [mau.KENH_TINH.index(k) - 1 for k in dataset.KENH_DAT]
    assert torch.all(b["tinh"][dat] == 0) and torch.any(b["tinh"][0] != 0)
    assert torch.all(b["quan_trac"] == 0) and torch.all(b["du_bao"] == 0) and torch.all(b["s1"] == 0)
    assert b["bo"] == "dat,khi_tuong,s1"


def test_bo_cn_chi_dien_0_kenh_cn_giu_dat_tho(monkeypatch, tmp_path):
    # Thí nghiệm "CN so với đất thô" (thiet_ke_paper mục bỏ thành phần): chỉ bỏ CN, giữ sét/cát/dung trọng
    _gia(monkeypatch, tmp_path)  # gọi lại _gia trước mỗi lần đọc để hai lần rút cùng hạt giống
    goc = dataset.MauNgap("train")[0]
    _gia(monkeypatch, tmp_path)
    b = dataset.MauNgap("train", bo={"cn"})[0]
    cn = mau.KENH_TINH.index("cn") - 1
    assert torch.all(b["tinh"][cn] == 0) and torch.any(goc["tinh"][cn] != 0)
    khac = [i for i in range(b["tinh"].shape[0]) if i != cn]
    assert torch.equal(b["tinh"][khac], goc["tinh"][khac]) and torch.equal(b["s1"], goc["s1"])


def test_bo_dat_tho_chi_giu_cn(monkeypatch, tmp_path):
    # Vế còn lại của "CN so với đất thô": bỏ sét/cát/dung trọng, giữ CN và mọi kênh khác
    _gia(monkeypatch, tmp_path)
    goc = dataset.MauNgap("train")[0]
    _gia(monkeypatch, tmp_path)
    b = dataset.MauNgap("train", bo={"dat_tho"})[0]
    tho = [mau.KENH_TINH.index(k) - 1 for k in dataset.KENH_DAT if k != "cn"]
    assert torch.all(b["tinh"][tho] == 0) and torch.any(goc["tinh"][tho] != 0)
    khac = [i for i in range(b["tinh"].shape[0]) if i not in tho]
    assert torch.equal(b["tinh"][khac], goc["tinh"][khac])


def test_tang_cuong_lat_dong_bo_anh_va_nhan(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    ds = dataset.MauNgap("train", tang_cuong=True, seed=1)
    for i in range(4):
        b = ds[0]
        ngap = (b["nhan"] == 1)
        assert ngap.sum() == 200 and (b["nhan"] == 255).sum() == 5 * P  # chỉ đổi chỗ, không đổi số lượng
        k = torch.nonzero(ngap)[0]
        assert torch.isfinite(b["tinh"][:, k[0], k[1]]).all()


def test_khong_doc_tap_test(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        dataset.MauNgap("test_b")


def test_sampler_chi_cho_train_va_can_bang(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        dataset.lay_mau_can_bang(dataset.MauNgap("val"), seed=0)
    ds = dataset.MauNgap("train")
    s = list(dataset.lay_mau_can_bang(ds, ty_le_o_co_ngap=0.5, seed=0, so_mau=4000))
    co = ds.d.ty_le_ngap.to_numpy()[s] > 0
    assert abs(co.mean() - 0.5) < 0.03 and s == list(dataset.lay_mau_can_bang(ds, seed=0, so_mau=4000))


def test_chuan_hoa_chi_tinh_tren_train(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    doc = []
    goc = dataset.mau.doc_mau
    monkeypatch.setattr(dataset.mau, "doc_mau", lambda g, r, **k: doc.append(r.vai_tro) or goc(g, r, **k))
    j = dataset.tinh_chuan_hoa(so_mau=3, seed=0)
    assert set(doc) == {"train"} and len(j["tinh_tb"]) == 11 and len(j["qt_tb"]) == 10
    assert json.loads(dataset.F_CHUAN_HOA.read_text())["so_mau"] == 3


def test_chuan_hoa_cot_rong_thanh_0_1(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path)
    goc = dataset.mau.doc_mau

    def glofas_rong(g, r, **k):
        m = goc(g, r, **k)
        m["quan_trac"][:, 9] = np.nan
        return m

    monkeypatch.setattr(dataset.mau, "doc_mau", glofas_rong)
    j = dataset.tinh_chuan_hoa(so_mau=3, seed=0)
    assert j["qt_tb"][9] == 0.0 and j["qt_ls"][9] == 1.0 and j["cot_rong"] == ["glofas_q"]
    assert "NaN" not in dataset.F_CHUAN_HOA.read_text()


def test_co_s2_cap_anh_chuan_hoa_prithvi_theo_nam_ngay_t(monkeypatch, tmp_path):
    # B4: ảnh S2 mùa khô của năm ngày t, chuẩn hóa theo trung bình/độ lệch HLS của Prithvi-EO-2.0 (terratorch
    # prithvi_vit.PRITHVI_V2_MEAN/STD, đơn vị phản xạ x 10000); pixel NaN về 0 (tức bằng trung bình)
    _gia(monkeypatch, tmp_path)
    goi = []

    def gia_s2(tile_id, o_utm, nam):
        goi.append((tile_id, nam))
        a = np.full((6, P, P), 0.2, np.float32)
        a[:, 0, 0] = np.nan
        return a
    monkeypatch.setattr(dataset.s2, "anh_mua_kho", gia_s2)
    assert "s2" not in dataset.MauNgap("train")[0]
    b = dataset.MauNgap("train", co_s2=True)[0]
    assert goi == [("huong_1_1", 2020)] and b["s2"].shape == (6, P, P)
    assert abs(float(b["s2"][3, 5, 5]) - (2000 - 2734) / 1850) < 1e-5 and float(b["s2"][0, 0, 0]) == 0


def test_mau_cho_mo_hinh_trung_dataset_khi_khong_tang_cuong(monkeypatch, tmp_path):
    # Task 16 chấm test qua danh_gia (du_doan nhận mẫu thô từ mau.doc_mau): chuẩn hóa phải trùng hệt lúc train
    _gia(monkeypatch, tmp_path)
    ds = dataset.MauNgap("train", bo={"cn"})
    a = ds[0]
    _gia(monkeypatch, tmp_path)
    r = ds.d.iloc[0]
    m = dataset.mau.doc_mau(None, r)
    b = dataset.mau_cho_mo_hinh(m, r, bo={"cn"})
    for k in ("s1", "dem", "tinh", "quan_trac", "du_bao", "mat_na_du_bao"):
        assert b[k].shape == (1, *a[k].shape) and torch.equal(b[k][0], a[k]), k
    assert int(b["L"][0]) == a["L"] and abs(float(b["khoang_cach_s1"][0]) - a["khoang_cach_s1"]) < 1e-7
