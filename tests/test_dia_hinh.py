import numpy as np

from floodlab import dia_hinh


def _dem_ranh(n=60, doc=0.01, pixel=20.0):
    # mặt nghiêng về phía nam, một rãnh dọc giữa sâu hơn hai bên, nước chảy về rãnh rồi ra mép dưới
    y, x = np.mgrid[0:n, 0:n]
    return (n - y) * pixel * doc + np.abs(x - n // 2) * pixel * 0.05


def test_hand_mat_phang_nghieng():
    dem = _dem_ranh()
    h = dia_hinh.hand(dem, 20.0, nguong_dong_km2=0.01)
    giua = 30
    assert np.allclose(h[5:-5, giua], 0, atol=0.5)
    hang = h[30, giua:giua + 20]
    assert np.all(np.diff(hang) >= -1e-6) and hang[-1] > hang[0]


def test_do_doc_mat_phang():
    y, x = np.mgrid[0:20, 0:20]
    dem = x * 20.0 * 0.10  # dốc 10% theo trục x
    s = dia_hinh.do_doc(dem, 20.0)
    assert abs(float(np.median(s[2:-2, 2:-2])) - 5.71) < 0.01


def test_hand_lap_lo_bang_kenh_gan_nhat():
    dem = np.array([[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]])
    kenh = np.zeros((3, 3), bool)
    kenh[:, 0] = True
    hand = np.array([[0, 1, np.nan], [0, np.nan, 2], [0, 1, 2]], float)
    out = dia_hinh.lap_hand(hand, dem, kenh)
    assert not np.isnan(out).any() and out[0, 2] == 2.0 and out[1, 1] == 1.0
