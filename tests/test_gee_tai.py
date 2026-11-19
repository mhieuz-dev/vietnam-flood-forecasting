import numpy as np
import pytest

from floodlab import gee_tai


class _GiaGee:
    def __init__(self):
        self.so_lan = 0

    def __call__(self, req):
        self.so_lan += 1
        return np.ones((4, 4), dtype=[("b1", "<f4")])


def test_tai_lai_bo_qua_file_da_co(tmp_path, monkeypatch):
    gia = _GiaGee()
    monkeypatch.setattr(gee_tai, "_compute_pixels", gia)
    p = tmp_path / "o.npy"
    gee_tai.tai_luoi(None, (0, 0), 4, 4, p)
    gee_tai.tai_luoi(None, (0, 0), 4, 4, p)
    assert gia.so_lan == 1 and p.exists()


def test_ghi_nguyen_tu(tmp_path, monkeypatch):
    def hong(req):
        raise KeyboardInterrupt

    monkeypatch.setattr(gee_tai, "_compute_pixels", hong)
    p = tmp_path / "o.npy"
    with pytest.raises(KeyboardInterrupt):
        gee_tai.tai_luoi(None, (0, 0), 4, 4, p)
    assert not p.exists() and not list(tmp_path.glob("*.tmp*"))


def test_thu_lai_khi_429(tmp_path, monkeypatch):
    lan = {"n": 0}

    def loi_roi_ok(req):
        lan["n"] += 1
        if lan["n"] < 3:
            raise gee_tai.ee.ee_exception.EEException("Too Many Requests (429)")
        return np.zeros((4, 4), dtype=[("b1", "<f4")])

    monkeypatch.setattr(gee_tai, "_compute_pixels", loi_roi_ok)
    monkeypatch.setattr(gee_tai.time, "sleep", lambda s: None)
    gee_tai.tai_luoi(None, (0, 0), 4, 4, tmp_path / "o.npy")
    assert lan["n"] == 3
