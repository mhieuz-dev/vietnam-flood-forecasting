import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from floodlab import danh_gia, mau
from floodlab.su_kien import SuKien

P = 224


def _gia(monkeypatch, tmp_path, goi_test):
    nhan = np.zeros((P, P), np.uint8)
    nhan[:50] = 1
    nhan[200:] = 255
    tinh = np.zeros((len(mau.KENH_TINH), P, P), np.float32)
    tinh[mau.KENH_TINH.index("worldcover")] = 40
    dong = pd.DataFrame({
        "tile_id": ["huong_1_1", "huong_1_2", "huong_1_1"], "luu_vuc": "huong",
        "ngay_t": [date(2020, 10, 13), date(2020, 10, 13), date(2025, 10, 20)], "L": [1, 2, 1],
        "vai_tro": ["val", "val", "test_b"], "loai_ngay": ["lu", "lu", "test_b"], "co_ngap_moi": True,
        "i_o": 0, "i_s1": 0, "i_nhan": 0, "i_mau": [0, 1, 2]})
    monkeypatch.setattr(danh_gia, "LUU_VUC", ("huong",))
    monkeypatch.setattr(danh_gia.mau, "doc_chi_muc", lambda lv: dong)
    monkeypatch.setattr(danh_gia.mau, "mo_zarr", lambda lv: None)
    monkeypatch.setattr(danh_gia.mau, "doc_mau", lambda g, r, **k: {
        "tinh": tinh, "s1": np.zeros((2, P, P), np.float32), "nhan": nhan,
        "nhan_trang_thai": np.zeros((P, P), np.uint8), "quan_trac": np.zeros((31, 10), np.float32),
        "du_bao": np.zeros((3, 2), np.float32), "mat_na": np.array([True, False, False])})

    def doc_test(ten, ly_do):
        goi_test.append((ten, ly_do))
        return pd.DataFrame({"tile_id": ["huong_1_1"], "ngay_t": ["2025-10-20"], "vai_tro": [ten]})

    monkeypatch.setattr(danh_gia.chia_tap, "doc_test", doc_test)
    monkeypatch.setattr(danh_gia, "doc_su_kien", lambda: [
        SuKien("huong_20201010", ("huong",), date(2020, 10, 10), date(2020, 10, 20), "train", (), "GFM"),
        SuKien("test_b_mien_trung_2025", ("huong",), date(2025, 10, 1), date(2025, 11, 30), "test_b", (), "EMS")])
    monkeypatch.setattr(danh_gia, "DIR_DANH_GIA", tmp_path / "danh_gia")
    monkeypatch.setattr(danh_gia, "DIR_DU_DOAN", tmp_path / "du_doan")
    return nhan


def test_du_doan_hoan_hao_va_test_doc_qua_cua_khoa(monkeypatch, tmp_path):
    goi = []
    nhan = _gia(monkeypatch, tmp_path, goi)
    hoan_hao = lambda m: (m["nhan"] == 1).astype(np.float32)
    p = danh_gia.danh_gia(hoan_hao, "val", "gfs", "thu", B=20)
    j = json.loads(p.read_text())
    assert j["so_mau"] == 2 and j["tong"]["csi"] == 1.0 and j["ngap_moi"]["csi"] == 1.0
    assert set(j["theo_L"]) == {"1", "2"} and "huong_20201010" in j["theo_tran"] and "40" in j["theo_lop_phu"]
    assert goi == []  # val không đi qua cửa test
    p = danh_gia.danh_gia(hoan_hao, "test_b", "gfs", "thu", B=20)
    j = json.loads(p.read_text())
    assert goi and goi[0][0] == "test_b" and j["so_mau"] == 1
    assert j["nguong_toi_uu"]["nguon"] == "val" and "test_b_mien_trung_2025" in j["theo_tran"]
    assert (tmp_path / "du_doan" / "thu_test_b_gfs_dem.parquet").exists()


def test_nguong_cho_test_phai_co_ket_qua_val(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path, [])
    with pytest.raises(FileNotFoundError):
        danh_gia.danh_gia(lambda m: np.zeros((P, P), np.float32), "test_b", "gfs", "chua_co_val", B=20)


def test_mau_test_ngoai_khoa_bao_loi(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path, [])
    monkeypatch.setattr(danh_gia.chia_tap, "doc_test", lambda ten, ly_do: pd.DataFrame(
        {"tile_id": ["huong_9_9"], "ngay_t": ["2025-10-20"], "vai_tro": [ten]}))
    danh_gia.danh_gia(lambda m: np.zeros((P, P), np.float32), "val", "gfs", "x", B=20)
    with pytest.raises(RuntimeError):
        danh_gia.danh_gia(lambda m: np.zeros((P, P), np.float32), "test_b", "gfs", "x", B=20)


def test_du_doan_nhan_chi_muc_cua_mau(monkeypatch, tmp_path):
    _gia(monkeypatch, tmp_path, [])
    thay = []
    danh_gia.danh_gia(lambda m: thay.append(m["chi_muc"]["tile_id"]) or np.zeros((P, P), np.float32), "val", "gfs", "y", B=5)
    assert thay == ["huong_1_1", "huong_1_2"]


def test_nguong_lay_tu_val_cua_luot_va_nguon_khac(monkeypatch, tmp_path):
    # Phép thử đất (ten khác) và mưa ECMWF/GSMaP (không có val cùng nguồn) phải dùng ngưỡng val GFS của chính mô hình
    _gia(monkeypatch, tmp_path, [])
    f = lambda m: np.full((P, P), 0.3, np.float32)
    danh_gia.danh_gia(f, "val", "gfs", "goc", B=20)
    p = danh_gia.danh_gia(f, "test_b", "ecmwf", "goc__nhieu", B=20, ten_nguong="goc", nguon_nguong="gfs",
                          luu_xac_suat=False)
    j = json.loads(p.read_text())
    assert p.name == "goc__nhieu_test_b_ecmwf.json" and j["nguong_toi_uu"]["file"] == "goc_val_gfs.json"
    assert not (tmp_path / "du_doan" / "goc__nhieu_test_b_ecmwf.zarr").exists()


def test_vai_tro_lobo_chi_lay_mau_train_cua_luu_vuc_bi_bo(monkeypatch, tmp_path):
    # LOBO (spec: kỹ năng trên vùng chưa thấy): chấm trên mẫu train của lưu vực bị bỏ khỏi lúc train, không qua cửa test
    goi = []
    _gia(monkeypatch, tmp_path, goi)
    dong = pd.DataFrame({
        "tile_id": ["huong_1_1", "ca_1_1", "huong_1_2", "huong_1_1"], "luu_vuc": ["huong", "ca", "huong", "huong"],
        "ngay_t": [date(2020, 10, 13)] * 3 + [date(2025, 10, 20)], "L": 1,
        "vai_tro": ["train", "train", "val", "test_b"], "loai_ngay": "lu", "co_ngap_moi": True,
        "i_o": 0, "i_s1": 0, "i_nhan": 0, "i_mau": [0, 1, 2, 3]})
    monkeypatch.setattr(danh_gia, "LUU_VUC", ("huong", "ca"))
    monkeypatch.setattr(danh_gia.mau, "doc_chi_muc", lambda lv: dong[dong.luu_vuc == lv])
    f = lambda m: np.zeros((P, P), np.float32)
    danh_gia.danh_gia(f, "val", "gfs", "lobo_x", B=20)
    p = danh_gia.danh_gia(f, "lobo_huong", "gfs", "lobo_x", B=20, luu_xac_suat=False)
    j = json.loads(p.read_text())
    assert j["so_mau"] == 1 and goi == [] and p.name == "lobo_x_lobo_huong_gfs.json"
