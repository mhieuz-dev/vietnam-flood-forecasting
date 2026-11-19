from datetime import date, timedelta

import numpy as np

from floodlab import s1


def _gia(tmp_path, monkeypatch, ngay):
    monkeypatch.setattr(s1, "THU_MUC", tmp_path)
    d = tmp_path / "o_1"
    d.mkdir()
    for n in ngay:
        np.save(d / f"{n:%Y%m%d}.npy", np.full((2, 4, 4), n.day, np.float16))


def test_anh_trang_thai_cung_ngay_t0_duoc_phep_ngay_t_bi_cam(tmp_path, monkeypatch):
    t0 = date(2020, 10, 12)
    _gia(tmp_path, monkeypatch, [date(2020, 10, 1), t0, date(2020, 10, 13)])
    for L in (1, 2, 3):
        t = t0 + timedelta(days=L)
        ngay, mang = s1.anh_trang_thai("o_1", t0)
        assert ngay <= t0 < t and mang.shape == (2, 4, 4)
    assert s1.anh_trang_thai("o_1", t0)[0] == t0
    assert s1.anh_trang_thai("o_1", date(2020, 10, 11))[0] == date(2020, 10, 1)


def test_khong_co_anh_trong_30_ngay_tra_none(tmp_path, monkeypatch):
    _gia(tmp_path, monkeypatch, [date(2020, 9, 1)])
    assert s1.anh_trang_thai("o_1", date(2020, 10, 12)) is None


def test_ngay_trang_thai_can_tai_luon_truoc_ngay_muc_tieu():
    co = [date(2020, 10, d) for d in (1, 5, 9, 12, 13, 20)]
    can = s1.ngay_trang_thai_can(co, [date(2020, 10, 13)])
    # t=13: L=1 -> t0=12 lấy 12; L=2 -> t0=11 lấy 9; L=3 -> t0=10 lấy 9
    assert can == {date(2020, 10, 12), date(2020, 10, 9)}
    assert all(n < date(2020, 10, 13) for n in can)
    assert s1.ngay_trang_thai_can(co, [date(2020, 12, 30)]) == set()  # không ảnh nào trong 30 ngày
