from datetime import date, timedelta

import numpy as np

from floodlab import du_bao, khi_tuong


def test_rh_bao_hoa():
    assert abs(float(khi_tuong.do_am_tuong_doi(np.array(25.0), np.array(25.0))) - 100) < 0.01
    assert 50 < float(khi_tuong.do_am_tuong_doi(np.array(30.0), np.array(20.0))) < 60


def test_api_hang_so():
    a = khi_tuong.api(np.full(400, 10.0), k=0.9)
    assert abs(a[-1] - 100) < 1


def test_khung_quan_trac_khong_vuot_t0():
    t0 = date(2020, 10, 12)
    ngay = [t0 - timedelta(days=200) + timedelta(days=i) for i in range(260)]
    chuoi = {n: float(n.toordinal()) for n in ngay}
    chuoi[t0 + timedelta(days=1)] = 1e9  # giá trị gài ở ngày sau t0
    cua_so = khi_tuong.cua_so_quan_trac(t0)
    assert cua_so[0] == t0 - timedelta(days=30) and cua_so[-1] == t0 and len(cua_so) == 31
    gia_tri = [chuoi[n] for n in cua_so]
    assert max(gia_tri) < 1e9


def test_mat_na_theo_L():
    assert du_bao.mat_na(1).tolist() == [True, False, False]
    assert du_bao.mat_na(2).tolist() == [True, True, False]
    assert du_bao.mat_na(3).tolist() == [True, True, True]


def test_du_bao_lay_chu_ky_t0_cong_1():
    assert du_bao.ngay_chu_ky(date(2020, 10, 12)) == date(2020, 10, 13)


def test_mua_ngay_tu_tich_luy():
    # tích lũy 0-24, 0-48, 0-72 giờ -> mưa từng ngày
    tl = np.array([[[10.0]], [[25.0]], [[25.0]]])
    assert du_bao.mua_ngay_tu_tich_luy(tl)[:, 0, 0].tolist() == [10.0, 15.0, 0.0]


def test_o_luoi_hue():
    assert khi_tuong.o_luoi(107.58, 16.46, 0.1) == (95, 105)
    assert khi_tuong.o_luoi(107.58, 16.46, 0.25) == (38, 42)


def test_khung_du_bao_mat_na_va_dien_0(tmp_path, monkeypatch):
    t0 = date(2020, 10, 12)
    luoi = np.zeros((3, 73, 53), np.float32)
    luoi[:, 38, 42] = [10, 20, 30]
    monkeypatch.setattr(du_bao, "doc_gfs", lambda d: luoi if d == date(2020, 10, 13) else None)
    monkeypatch.setattr(du_bao, "_vi_tri", lambda tile_id: ((38, 42), np.zeros((73, 53), bool) | (np.arange(53) == 42)[None, :] & (np.arange(73) == 38)[:, None]))
    gt, mn = du_bao.khung_du_bao("huong_1_1", t0, 2, "gfs")
    assert mn.tolist() == [True, True, False]
    assert gt[:, 0].tolist() == [10, 20, 0] and gt[:, 1].tolist() == [10, 20, 0]


def test_chuoi_gsmap_chi_thay_cot_mua_va_api(tmp_path, monkeypatch):
    t0 = date(2024, 9, 7)
    ngay = np.array([(t0 - timedelta(days=40) + timedelta(days=i)).toordinal() for i in range(45)])
    gt = np.ones((1, 45, 10), np.float32)
    gs = np.full((1, 45, 3), 7.0, np.float32)
    np.savez_compressed(tmp_path / "hong_thai_binh.npz", ngay=ngay, tile_id=np.array(["hong_thai_binh_1_1"]),
                        gia_tri=gt, gsmap=gs)
    monkeypatch.setattr(khi_tuong, "DIR_CHUOI", tmp_path)
    khi_tuong._chuoi.cache_clear()
    out = khi_tuong.chuoi_quan_trac("hong_thai_binh_1_1", t0, "gsmap_nrt")
    assert out.shape == (31, 10)
    assert (out[:, [0, 1, 8]] == 7).all() and (np.delete(out, [0, 1, 8], axis=1) == 1).all()
    assert (khi_tuong.chuoi_quan_trac("hong_thai_binh_1_1", t0)[:, 0] == 1).all()
    gs[0, 40, 0] = np.nan  # thiếu GSMaP ngày t0 (chỉ số 40) thì loại mẫu
    np.savez_compressed(tmp_path / "hong_thai_binh.npz", ngay=ngay, tile_id=np.array(["hong_thai_binh_1_1"]),
                        gia_tri=gt, gsmap=gs)
    khi_tuong._chuoi.cache_clear()
    assert khi_tuong.chuoi_quan_trac("hong_thai_binh_1_1", t0, "gsmap_nrt") is None
    khi_tuong._chuoi.cache_clear()


def test_khung_du_bao_oracle_lay_imerg_cac_ngay_sau_t0(tmp_path, monkeypatch):
    t0 = date(2024, 9, 7)
    ngay = np.array([(t0 + timedelta(days=i)).toordinal() for i in range(4)])
    gt = np.zeros((1, 4, 10), np.float32)
    gt[0, :, 0], gt[0, :, 1] = [0, 11, 22, 33], [0, 1, 2, 3]
    np.savez_compressed(tmp_path / "hong_thai_binh.npz", ngay=ngay, tile_id=np.array(["hong_thai_binh_1_1"]),
                        gia_tri=gt, gsmap=np.full((1, 4, 3), np.nan, np.float32))
    monkeypatch.setattr(khi_tuong, "DIR_CHUOI", tmp_path)
    khi_tuong._chuoi.cache_clear()
    g, mn = du_bao.khung_du_bao("hong_thai_binh_1_1", t0, 2, "imerg_final")
    khi_tuong._chuoi.cache_clear()
    assert g[:, 0].tolist() == [11, 22, 0] and g[:, 1].tolist() == [1, 2, 0]


def test_khung_du_bao_thieu_cache_tra_none_khong_tai_mang(tmp_path, monkeypatch):
    def cam_tai(*a, **k):
        raise AssertionError("lắp mẫu không được tải GFS qua mạng")

    monkeypatch.setattr(du_bao, "THU_MUC_GFS", tmp_path)
    monkeypatch.setattr(du_bao, "_ncar_tich_luy", cam_tai)
    monkeypatch.setattr(du_bao, "_aws_tich_luy", cam_tai)
    assert du_bao.khung_du_bao("huong_1_1", date(2016, 9, 25), 1, "gfs") is None


def test_ecmwf_doi_met_sang_mm_va_mua_ngay(tmp_path, monkeypatch):
    # ECMWF open data: tp tích lũy (m) từ đầu chu kỳ 00Z tới 24/48/72 h -> mưa ngày mm, cache riêng thư mục ecmwf
    monkeypatch.setattr(du_bao, "THU_MUC_ECMWF", tmp_path)
    tl = np.zeros((3, 73, 53), np.float32)
    tl[:, 0, 0] = [0.010, 0.025, 0.030]
    monkeypatch.setattr(du_bao, "_ecmwf_tich_luy", lambda d: tl)
    m = du_bao.tai_ecmwf(date(2024, 9, 10))
    assert np.allclose(m[:, 0, 0], [10, 15, 5]) and (tmp_path / "20240910.npy").exists()
    assert du_bao.doc_ecmwf(date(2024, 9, 10)) is not None and du_bao.doc_ecmwf(date(2024, 9, 11)) is None


def test_khung_du_bao_ecmwf_chi_doc_cache(tmp_path, monkeypatch):
    luoi = np.zeros((3, 73, 53), np.float32)
    luoi[:, 38, 42] = [7, 8, 9]
    monkeypatch.setattr(du_bao, "doc_ecmwf", lambda d: luoi if d == date(2024, 9, 10) else None)
    monkeypatch.setattr(du_bao, "_vi_tri", lambda tile_id: ((38, 42), (np.arange(53) == 42)[None, :] & (np.arange(73) == 38)[:, None]))
    gt, mn = du_bao.khung_du_bao("hong_1_1", date(2024, 9, 9), 3, "ecmwf")
    assert gt[:, 0].tolist() == [7, 8, 9] and mn.all()
    assert du_bao.khung_du_bao("hong_1_1", date(2024, 9, 1), 3, "ecmwf") is None
