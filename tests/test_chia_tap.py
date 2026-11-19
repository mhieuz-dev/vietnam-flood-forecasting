from datetime import date, timedelta

import pandas as pd
import pytest

from floodlab import chia_tap
from floodlab.su_kien import doc_su_kien, vai_tro_cua_ngay


def test_doc_su_kien_co_hai_tap_test():
    vai_tro = {sk.vai_tro for sk in doc_su_kien()}
    assert {"test_a", "test_b"} <= vai_tro


def test_khong_trung_ngay_train_test():
    for luu_vuc in ("hong_thai_binh", "huong", "gianh"):
        d = date(2024, 1, 1)
        while d <= date(2025, 12, 31):
            assert vai_tro_cua_ngay(luu_vuc, d) != "train"
            d += timedelta(days=1)


def test_cua_so_test_dung():
    assert vai_tro_cua_ngay("hong_thai_binh", date(2024, 9, 10)) == "test_a"
    assert vai_tro_cua_ngay("huong", date(2025, 10, 20)) == "test_b"
    assert vai_tro_cua_ngay("huong", date(2024, 9, 10)) is None
    assert vai_tro_cua_ngay("hong_thai_binh", date(2025, 10, 20)) is None
    assert vai_tro_cua_ngay("huong", date(2020, 10, 13)) == "train"


def _tao_tap(tmp_path, monkeypatch):
    thu_muc = tmp_path / "nhan"
    thu_muc.mkdir()
    dong = []
    for i, vai_tro in enumerate(["test_a", "test_a", "test_b"]):
        f = thu_muc / f"{i}.npy"
        f.write_bytes(bytes([i]) * 100)
        dong.append({"tile_id": f"o_{i}", "ngay_t": f"2024-09-1{i}", "vai_tro": vai_tro, "file_nhan": str(f)})
    monkeypatch.setattr(chia_tap, "NHAT_KY", tmp_path / "nhat_ky_test.log")
    khoa = tmp_path / "khoa.json"
    chia_tap.khoa_tap_test(pd.DataFrame(dong), path=khoa)
    return khoa, thu_muc


def test_doc_test_bao_loi_khi_bam_doi(tmp_path, monkeypatch):
    khoa, thu_muc = _tao_tap(tmp_path, monkeypatch)
    f = thu_muc / "0.npy"
    b = bytearray(f.read_bytes())
    b[0] ^= 1
    f.write_bytes(bytes(b))
    with pytest.raises(RuntimeError, match="tap test da thay doi"):
        chia_tap.doc_test("test_a", "kiem thu", path=khoa)


def test_doc_test_ghi_nhat_ky(tmp_path, monkeypatch):
    khoa, _ = _tao_tap(tmp_path, monkeypatch)
    ds = chia_tap.doc_test("test_a", "lan 1", path=khoa)
    assert len(ds) == 2
    chia_tap.doc_test("test_b", "lan 2", path=khoa)
    dong = (tmp_path / "nhat_ky_test.log").read_text(encoding="utf-8").splitlines()
    assert len(dong) == 2
    assert "test_a" in dong[0] and "lan 1" in dong[0]


def test_git_hash_lay_tu_bien_moi_truong_khi_khong_co_git(monkeypatch, tmp_path):
    # Chấm trên Kaggle không có thư mục .git: kết quả vẫn phải ghi đúng commit của mã đã đẩy lên
    monkeypatch.setattr(chia_tap, "ROOT", tmp_path)
    monkeypatch.setenv("FLOODLAB_GIT", "abc1234")
    assert chia_tap._git_hash() == "abc1234"
