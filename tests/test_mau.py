from datetime import date

import numpy as np
import pandas as pd
import pytest

from floodlab import mau

T = date(2020, 10, 13)


def _gia(monkeypatch, s1=True, gfs=True, nhan_tt=None):
    nhan = np.zeros((4, 4), np.uint8)
    nhan[0, 0] = 1
    monkeypatch.setattr(mau, "_doc_nhan", lambda tid, n: nhan if n == T else (nhan_tt if nhan_tt is not None else np.zeros((4, 4), np.uint8)))
    monkeypatch.setattr(mau, "_loai_tru", lambda tid: np.zeros((4, 4), bool))
    monkeypatch.setattr(mau, "_ngay_co_nhan", lambda tid: [date(2020, 9, 1), date(2020, 10, 11), T])
    monkeypatch.setattr(mau, "_bang_ty_le", lambda lv: pd.DataFrame({"ngay": pd.to_datetime([]), "co_loai_tru": []}))
    monkeypatch.setattr(mau.s1, "anh_trang_thai",
                        lambda tid, t0: (date(2020, 10, 9), np.zeros((2, 4, 4), np.float16)) if s1 else None)
    monkeypatch.setattr(mau.khi_tuong, "chuoi_quan_trac", lambda tid, t0, nguon_mua="imerg_final": np.zeros((31, 10), np.float32))
    monkeypatch.setattr(mau.du_bao, "khung_du_bao",
                        lambda tid, t0, L, nguon: (np.zeros((3, 2), np.float32), np.array([True, False, False]))
                        if (gfs or nguon != "gfs") else None)


def test_thieu_s1_bi_loai_co_ly_do(monkeypatch):
    _gia(monkeypatch, s1=False)
    ket, ly_do = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert ket is None and ly_do == "khong_co_s1_30_ngay"


def test_thieu_gfs_bi_loai(monkeypatch):
    _gia(monkeypatch, gfs=False)
    ket, ly_do = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert ket is None and ly_do == "thieu_du_bao_gfs"


def test_vai_tro_val_la_gianh():
    assert mau.vai_tro("gianh", date(2020, 10, 1), "lu") == "val"
    assert mau.vai_tro("gianh", date(2020, 10, 1), "khong_lu") == "val"
    assert mau.vai_tro("huong", date(2020, 10, 1), "lu") == "train"
    assert mau.vai_tro("gianh", date(2025, 10, 20), "test_b") == "test_b"


def test_khong_mau_test_trong_train():
    with pytest.raises(ValueError):
        mau.vai_tro("huong", date(2025, 10, 20), "lu")  # ngày trong cửa sổ test không được làm train
    with pytest.raises(ValueError):
        mau.vai_tro("huong", date(2020, 10, 20), "test_b")


def test_ngap_moi(monkeypatch):
    _gia(monkeypatch)  # trạng thái khô, mục tiêu có 1 pixel ngập -> ngập mới
    ket, _ = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert ket["chi_muc"]["co_ngap_moi"]
    tt = np.zeros((4, 4), np.uint8)
    tt[0, 0] = 1  # pixel đó đã ngập ở ảnh trạng thái -> không phải ngập mới
    _gia(monkeypatch, nhan_tt=tt)
    ket, _ = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert not ket["chi_muc"]["co_ngap_moi"]
    assert ket["chi_muc"]["t0"] == date(2020, 10, 12) and ket["chi_muc"]["khoang_cach_s1_ngay"] == 4  # spec §5: từ ảnh S1 đến t


def test_nhan_trang_thai_la_ban_do_gfm_moi_nhat_truoc_t0(monkeypatch):
    _gia(monkeypatch)  # S1 trạng thái 10/10/09 không có nhãn; nhãn GFM mới nhất <= t0 = 12/10 là 11/10
    ket, _ = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert ket["chi_muc"]["ngay_nhan_trang_thai"] == date(2020, 10, 11)
    assert ket["chi_muc"]["ngay_s1_trang_thai"] == date(2020, 10, 9)
    monkeypatch.setattr(mau, "_ngay_co_nhan", lambda tid: [date(2020, 9, 1), T])  # 9/1 cách t0 quá 30 ngày
    ket, _ = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert ket["chi_muc"]["ngay_nhan_trang_thai"] is None and (ket["nhan_trang_thai"] == 255).all()


def test_ngap_moi_khong_tinh_pixel_trang_thai_chua_biet(monkeypatch):
    tt = np.zeros((4, 4), np.uint8)
    tt[0, 0] = 255  # pixel ngập ở t nhưng trạng thái không quan sát được -> không biết có phải ngập mới
    _gia(monkeypatch, nhan_tt=tt)
    ket, _ = mau.mau_mot("huong_1_1", T, "lu", 1)
    assert not ket["chi_muc"]["co_ngap_moi"]


def test_ghi_zarr_mang_nho_gop_khoi_va_khoa_phang(tmp_path):
    import zarr

    g = zarr.open_group(tmp_path / "x.zarr", mode="w")
    nho, lon = mau._Ghi(g, "du_bao", (2, 3, 2), "float32"), mau._Ghi(g, "nhan", (224, 224), "uint8")
    for i in range(100):
        nho.them(np.full((2, 3, 2), i, np.float32))
        lon.them(np.full((224, 224), i, np.uint8))
    nho.xa(), lon.xa()
    assert g["du_bao"].chunks[0] == mau.KHOI_MANG_NHO and g["nhan"].chunks[0] == 1
    assert g["du_bao"][57, 1, 2, 1] == 57 and g["nhan"][99, 223, 223] == 99
    tep = [p for p in (tmp_path / "x.zarr").rglob("*") if p.is_file() and p.name != "zarr.json"]
    assert all(p.parent.parent == tmp_path / "x.zarr" for p in tep)  # x.zarr/{mang}/c.i.0..., không lồng thư mục
    assert len([p for p in tep if p.parent.name == "du_bao"]) == 1


def test_mat_na_gop_chi_ap_ngay_thieu_lop_loai_tru(monkeypatch):
    _gia(monkeypatch)
    bang = pd.DataFrame({"ngay": pd.to_datetime(["2016-10-01", "2020-10-12", "2021-10-12", "2020-10-13"]),
                         "co_loai_tru": [1, 1, 0, 1]})
    monkeypatch.setattr(mau, "_bang_ty_le", lambda lv: bang)
    assert mau._can_mat_na_gop("huong_1_1", date(2016, 10, 1))  # 2016-2017: sản phẩm cũ, gộp cho chắc
    assert mau._can_mat_na_gop("huong_1_1", date(2021, 10, 12))  # sản phẩm thiếu lớp loại trừ
    assert not mau._can_mat_na_gop("huong_1_1", date(2020, 10, 12))
    lt = np.zeros((4, 4), bool)
    lt[1, 1] = True
    monkeypatch.setattr(mau, "_loai_tru", lambda tid: lt)
    ket, _ = mau.mau_mot("huong_1_1", T, "lu", 1)  # T = 2020-10-13 có lớp loại trừ riêng -> không áp mặt nạ gộp
    assert ket["nhan"][1, 1] == 0


def test_doc_mau_lay_dung_lat_cat_theo_chi_so_va_nguon(tmp_path):
    import zarr

    g = zarr.open_group(tmp_path / "x.zarr", mode="w")
    P = 224
    ghi = {"tinh": mau._Ghi(g, "tinh", (len(mau.KENH_TINH), P, P), "float16"), "s1": mau._Ghi(g, "s1", (2, P, P), "float16"),
           "nhan_trang_thai": mau._Ghi(g, "nhan_trang_thai", (P, P), "uint8"), "nhan": mau._Ghi(g, "nhan", (P, P), "uint8"),
           "quan_trac": mau._Ghi(g, "quan_trac", (2, 31, 10), "float32"), "du_bao": mau._Ghi(g, "du_bao", (2, 3, 2), "float32"),
           "mat_na": mau._Ghi(g, "mat_na", (3,), "bool")}
    ghi["tinh"].them(np.full((12, P, P), 7, np.float16))
    for i in range(2):
        ghi["s1"].them(np.full((2, P, P), i, np.float16))
        ghi["nhan_trang_thai"].them(np.full((P, P), i, np.uint8))
        ghi["nhan"].them(np.full((P, P), 1 - i, np.uint8))
    for i in range(3):
        ghi["quan_trac"].them(np.stack([np.full((31, 10), i), np.full((31, 10), 10 + i)]).astype(np.float32))
        ghi["du_bao"].them(np.stack([np.full((3, 2), i), np.full((3, 2), 20 + i)]).astype(np.float32))
        ghi["mat_na"].them(np.array([True, i > 0, i > 1]))
    for w in ghi.values():
        w.xa()
    cm = pd.DataFrame({"i_o": [0, 0, 0], "i_s1": [0, 1, 1], "i_nhan": [1, 0, 0], "L": [1, 2, 3]}).assign(i_mau=[0, 1, 2])
    r = cm.iloc[2]
    m = mau.doc_mau(g, r, nguon_du_bao="imerg_final", nguon_quan_trac="gsmap_nrt")
    assert m["s1"].dtype == np.float32 and m["s1"][0, 0, 0] == 1 and m["tinh"][3, 5, 5] == 7
    assert m["nhan"][0, 0] == 1 and m["nhan_trang_thai"][0, 0] == 1
    assert m["quan_trac"][0, 0] == 12 and m["du_bao"][0, 0] == 22 and m["mat_na"].tolist() == [True, True, True]
    assert mau.doc_mau(g, cm.iloc[0])["du_bao"][0, 0] == 0  # mặc định gfs, imerg_final


def test_doc_mau_ecmwf_tinh_khung_luc_doc_voi_t0_bang_t_tru_L(monkeypatch):
    # ECMWF chỉ dùng lúc chấm test (không có trong zarr): khung tính khi đọc, t0 = t - L; thiếu cache thì báo lỗi
    import numpy as np
    from datetime import date
    from types import SimpleNamespace
    goi = []

    def khung(tile_id, t0, L, nguon):
        goi.append((tile_id, t0, L, nguon))
        return np.full((3, 2), 5.0, np.float32), np.array([True, True, False])
    monkeypatch.setattr(mau.du_bao, "khung_du_bao", khung)
    g = {"tinh": np.zeros((1, 12, 2, 2)), "s1": np.zeros((1, 2, 2, 2)), "nhan": np.zeros((1, 2, 2)),
         "nhan_trang_thai": np.zeros((1, 2, 2)), "quan_trac": np.zeros((1, 2, 31, 10)), "du_bao": np.zeros((1, 2, 3, 2)),
         "mat_na": np.array([[True, True, False]])}
    r = SimpleNamespace(i_o=0, i_s1=0, i_nhan=0, i_mau=0, tile_id="hong_1_1", ngay_t=date(2024, 9, 12), L=2)
    m = mau.doc_mau(g, r, nguon_du_bao="ecmwf")
    assert goi == [("hong_1_1", date(2024, 9, 10), 2, "ecmwf")] and m["du_bao"][0, 0] == 5.0
    monkeypatch.setattr(mau.du_bao, "khung_du_bao", lambda *a: None)
    import pytest
    with pytest.raises(RuntimeError):
        mau.doc_mau(g, r, nguon_du_bao="ecmwf")
