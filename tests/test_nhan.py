from datetime import date
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

from floodlab import gfm, su_kien
from floodlab.config import PIXEL_M, TILE_M, TILE_PX


def test_ma_nhan_hop_le():
    flood = np.array([[0, 1, 255, 1]], np.uint8)
    excl = np.array([[0, 0, 0, 1]], np.uint8)
    ref = np.array([[0, 0, 0, 0]], np.uint8)
    jrc = np.zeros((1, 4), bool)
    out = gfm.ma_hoa_nhan(flood, excl, ref, jrc)
    assert set(np.unique(out)) <= {0, 1, 255}
    assert out.tolist() == [[0, 1, 255, 255]]


def test_nuoc_thuong_xuyen_thanh_255():
    flood = np.array([[1, 0, 0]], np.uint8)
    excl = np.zeros((1, 3), np.uint8)
    ref = np.array([[1, 2, 0]], np.uint8)
    jrc = np.array([[False, False, True]])
    assert gfm.ma_hoa_nhan(flood, excl, ref, jrc).tolist() == [[255, 255, 255]]


def test_vung_loai_tru_thanh_255():
    out = gfm.ma_hoa_nhan(np.array([[1]], np.uint8), np.array([[1]], np.uint8),
                          np.array([[0]], np.uint8), np.zeros((1, 1), bool))
    assert out[0, 0] == 255


def test_chieu_lai_giu_vi_tri():
    # raster nguồn EPSG:32648, 10 m, chỉ một pixel = 1 tại tọa độ đã biết
    x0, y0 = 100 * TILE_M, 400 * TILE_M  # góc dưới-trái của ô
    o = box(x0, y0, x0 + TILE_M, y0 + TILE_M)
    src = np.zeros((1000, 1000), np.uint8)
    tf = Affine(10, 0, x0 - 500, 0, -10, y0 + TILE_M + 500)
    # pixel đích hàng 37, cột 15 phủ x0+300..320, đỉnh-740..760; tô kín khối 2 x 2 pixel nguồn 10 m tương ứng
    r0, c0 = int((tf.f - (y0 + TILE_M - 37 * PIXEL_M)) // 10), int((x0 + 15 * PIXEL_M - tf.c) // 10)
    src[r0:r0 + 2, c0:c0 + 2] = 1
    out = gfm.chieu_ve_o(src, tf, "EPSG:32648", o, nodata=255)
    assert out.shape == (TILE_PX, TILE_PX)
    assert np.argwhere(out == 1).tolist() == [[37, 15]]


def test_gop_cung_ngay_uu_tien_co_quan_sat():
    a = np.array([[255, 0, 1]], np.uint8)
    b = np.array([[1, 255, 0]], np.uint8)
    assert gfm.gop_cung_ngay([a, b]).tolist() == [[1, 0, 1]]


def test_chieu_vung_roi_cat_bang_chieu_tung_o():
    x0, y0 = 100 * TILE_M, 400 * TILE_M
    rng = np.random.default_rng(0)
    src = rng.integers(0, 2, (1200, 1200)).astype(np.uint8)
    tf = Affine(10, 0, x0 - 500, 0, -10, y0 + 2 * TILE_M + 500)
    b = (x0, y0, x0 + 2 * TILE_M, y0 + 2 * TILE_M)
    vung = gfm.chieu_ve_vung(src, tf, "EPSG:32648", b)
    o = box(x0 + TILE_M, y0, x0 + 2 * TILE_M, y0 + TILE_M)
    assert np.array_equal(gfm.cat_o(vung, b, o), gfm.chieu_ve_o(src, tf, "EPSG:32648", o))


def _bang(dong):
    return pd.DataFrame(dong, columns=["ngay", "loai", "pixel_hop_le", "ty_le_ngap"]).assign(
        ngay=lambda d: pd.to_datetime(d.ngay))


def test_nguong_tran_co_san_khi_trung_vi_kho_bang_0():
    d = _bang([("2020-02-01", "mua_kho", 100, 0.0)] * 20 + [("2020-03-01", "mua_kho", 100, 0.001)])
    assert su_kien.nguong_tran(d) == pytest.approx(su_kien.SAN_NGUONG_TRAN)


def test_nguong_tran_theo_phan_vi_95_mua_kho():
    d = _bang([("2020-02-01", "mua_kho", 100, x / 100) for x in range(1, 101)])
    assert su_kien.nguong_tran(d) == pytest.approx(max(2 * 0.505, 0.9505))


def test_tach_tran_gop_ngay_gan_va_bo_ngay_phu_it():
    d = _bang([
        ("2020-10-01", "mua_mua", 1000, 0.10),
        ("2020-10-12", "mua_mua", 1000, 0.20),   # cách 11 ngày: cùng trận
        ("2020-10-20", "mua_mua", 100, 0.90),    # phủ 10% < 25%: bỏ
        ("2020-12-01", "mua_mua", 1000, 0.05),   # cách 50 ngày: trận mới
        ("2020-12-05", "mua_mua", 1000, 0.001),  # dưới ngưỡng
        ("2021-03-01", "mua_kho", 9000, 0.0),    # sản phẩm thiếu lớp loại trừ: không được làm mốc phủ
    ])
    tran = su_kien.tach_tran(d, nguong=0.01)
    assert [(t["bat_dau"].isoformat(), t["ket_thuc"].isoformat(), t["so_ngay_anh"]) for t in tran] == [
        ("2020-10-01", "2020-10-12", 2), ("2020-12-01", "2020-12-01", 1)]


def test_tai_thu_lai_khi_mat_ket_noi(monkeypatch):
    import requests

    class TraLoi:
        status_code, ok = 200, True

        def raise_for_status(self):
            pass

    goi = []

    def gia_get(url, **kw):
        goi.append(url)
        if len(goi) < 3:
            raise (requests.exceptions.ReadTimeout if len(goi) == 1 else requests.exceptions.ConnectionError)()
        return TraLoi()

    monkeypatch.setattr(gfm.requests, "get", gia_get)
    monkeypatch.setattr(gfm.time, "sleep", lambda s: None)
    assert isinstance(gfm._get_thu_lai("https://x"), TraLoi)
    assert len(goi) == 3


def test_chon_ngay_muc_tieu_can_bang_va_co_dinh():
    d = _bang([(f"2020-{m:02d}-{k:02d}", "mua_mua", 1000, 0.2 if (m == 10 and k < 10) else 0.0)
               for m in (9, 10, 11) for k in range(1, 29, 3)]
              + [("2025-10-20", "test_b", 1000, 0.3), ("2021-02-01", "mua_kho", 1000, 0.0)])
    a = su_kien.chon_ngay_muc_tieu(d, nguong=0.01, seed=0)
    assert a == su_kien.chon_ngay_muc_tieu(d, nguong=0.01, seed=0)
    lu = [n for n, loai in a.items() if loai == "lu"]
    assert len(lu) == 3 and sum(v == "khong_lu" for v in a.values()) == 3
    assert a[date(2025, 10, 20)] == "test_b" and date(2021, 2, 1) not in a


def test_chon_ngay_muc_tieu_giu_ngay_phu_mot_phan():
    d = _bang([(f"2020-10-{k:02d}", "mua_mua", 1000, 0.2) for k in (1, 4, 7)]
              + [("2020-10-05", "mua_mua", 50, 0.0), ("2020-11-20", "mua_mua", 50, 0.0),
                 ("2025-10-20", "test_b", 50, 0.0)])
    a = su_kien.chon_ngay_muc_tieu(d, nguong=0.01, seed=0)
    assert a[date(2025, 10, 20)] == "test_b"  # ngày test phủ ít vẫn phải giữ
    assert a[date(2020, 10, 5)] == "lu"       # nằm trong trận, phủ ít vẫn là mẫu cho các ô được phủ
    assert a[date(2020, 11, 20)] == "khong_lu"


def test_san_pham_loi_may_chu_bi_bo_qua_con_loi_khac_van_bao(monkeypatch):
    import requests

    def nem(ma):
        def f(pid):
            r = requests.Response()
            r.status_code = ma
            raise requests.HTTPError(response=r)
        return f

    monkeypatch.setattr(gfm, "tai_lop", nem(500))
    assert gfm.tai_lop_hoac_bo("x") is None
    monkeypatch.setattr(gfm, "tai_lop", nem(401))
    with pytest.raises(requests.HTTPError):
        gfm.tai_lop_hoac_bo("x")


def test_zip_hong_thu_lai_roi_moi_bo(monkeypatch):
    import zipfile

    goi = []

    def zip_hong_lan_dau(pid):
        goi.append(pid)
        if len(goi) == 1:
            raise zipfile.BadZipFile("File is not a zip file")
        return {"ngap": 1}

    monkeypatch.setattr(gfm.time, "sleep", lambda s: None)
    monkeypatch.setattr(gfm, "tai_lop", zip_hong_lan_dau)
    assert gfm.tai_lop_hoac_bo("x") == {"ngap": 1}

    def luon_hong(pid):
        raise zipfile.BadZipFile("File is not a zip file")

    monkeypatch.setattr(gfm, "tai_lop", luon_hong)
    assert gfm.tai_lop_hoac_bo("x") is None


def test_bo_thang_cay_lua_chi_cat_ngay_train_song_hong():
    d = pd.DataFrame({"ngay": pd.to_datetime(["2017-02-10", "2017-03-10", "2017-07-01", "2017-09-16", "2024-09-12"]),
                      "loai": ["mua_kho", "mua_kho", "mua_mua", "mua_mua", "test_a"]})
    giu = su_kien.bo_thang_cay_lua(d, "hong_thai_binh")
    assert [n.strftime("%m-%d") for n in giu.ngay] == ["03-10", "09-16", "09-12"]
    assert len(su_kien.bo_thang_cay_lua(d, "huong")) == 5


def test_cua_so_co_dem_33_ngay_truoc_moi_cua_so_va_dat_sau_cung():
    import sys
    sys.path.insert(0, "scripts")
    from tai_nhan import cua_so

    cs = cua_so("huong")
    chinh = [c for c in cs if c[2] != "dem"]
    dem = [c for c in cs if c[2] == "dem"]
    assert len(dem) == len(chinh) and cs == chinh + dem
    assert (date(2016, 7, 30), date(2016, 9, 1), "dem") in dem
    assert all((den - tu).days == 33 and den in {c[0] for c in chinh} for tu, den, _ in dem)
