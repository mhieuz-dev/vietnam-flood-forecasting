from datetime import date

import numpy as np
import pytest

from floodlab import doi_chung


def test_b6_chep_trang_thai():
    nhan = np.array([[0, 1], [255, 1]], np.uint8)
    assert doi_chung.b6_giu_trang_thai(nhan).tolist() == [[0.0, 1.0], [0.0, 1.0]]


def test_b0_chi_dung_ngay_train(tmp_path, monkeypatch):
    o = tmp_path / "huong_1_1"
    o.mkdir()
    np.save(o / "20200101.npy", np.array([[1, 0], [255, 0]], np.uint8))
    np.save(o / "20200201.npy", np.array([[1, 1], [0, 0]], np.uint8))
    monkeypatch.setattr(doi_chung, "DIR_NHAN", tmp_path)
    monkeypatch.setattr(doi_chung, "DIR_B0", tmp_path / "_b0")
    truoc = doi_chung.b0_tan_suat("huong_1_1", dung_cache=False)
    np.save(o / "20251020.npy", np.ones((2, 2), np.uint8))  # ngày Test B ngập toàn bộ
    sau = doi_chung.b0_tan_suat("huong_1_1", dung_cache=False)
    assert np.array_equal(truoc, sau)
    assert sau.tolist() == [[1.0, 0.5], [0.0, 0.0]]  # pixel 255 không tính vào mẫu số


def test_b0_luu_cache_doc_lai_khi_nhan_tho_da_chuyen_di(tmp_path, monkeypatch):
    import shutil

    o = tmp_path / "huong_1_1"
    o.mkdir()
    np.save(o / "20200101.npy", np.array([[1, 0], [255, 0]], np.uint8))
    monkeypatch.setattr(doi_chung, "DIR_NHAN", tmp_path)
    monkeypatch.setattr(doi_chung, "DIR_B0", tmp_path / "_b0")
    a = doi_chung.b0_tan_suat("huong_1_1")
    shutil.rmtree(o)  # nhãn thô đã đưa lên Drive
    assert np.array_equal(doi_chung.b0_tan_suat("huong_1_1"), a)


def test_b1_hand_mua():
    hand = np.array([1.0, 1.0, 20.0])
    api = np.array([200.0, 10.0, 200.0])
    assert doi_chung.b1_hand_mua(hand, api, nguong_hand=5, nguong_api=100).tolist() == [1.0, 0.0, 0.0]


def test_nguong_b1_chon_tren_val():
    mau = [{"vai_tro": "val", "hand": np.array([1.0, 30.0]), "api": np.array([150.0, 150.0]),
            "nhan": np.array([1, 0], np.uint8)}]
    nh, na = doi_chung.chon_nguong_b1(mau)
    assert doi_chung.b1_hand_mua(mau[0]["hand"], mau[0]["api"], nh, na).tolist() == [1.0, 0.0]
    with pytest.raises(ValueError):
        doi_chung.chon_nguong_b1(mau + [{**mau[0], "vai_tro": "test_a"}])


def _mau_b2(vai_tro="train", seed=0):
    from floodlab.mau import KENH_TINH

    rng = np.random.default_rng(seed)
    tinh = np.zeros((len(KENH_TINH), 224, 224), np.float32)
    tinh[KENH_TINH.index("hand")] = rng.uniform(0, 20, (224, 224))
    nhan = (tinh[KENH_TINH.index("hand")] < 5).astype(np.uint8)
    nhan[:10] = 255
    return {"vai_tro": vai_tro, "tinh": tinh, "s1": rng.normal(-15, 3, (2, 224, 224)).astype(np.float32),
            "nhan": nhan, "nhan_trang_thai": np.zeros((224, 224), np.uint8),
            "quan_trac": np.ones((31, 10), np.float32), "du_bao": np.ones((3, 2), np.float32),
            "mat_na": np.array([True, False, False])}


def test_dac_trung_b2_khong_dung_du_bao_ngoai_han():
    m = _mau_b2()
    a = doi_chung.dac_trung_b2(m)
    m["du_bao"][1:] = 500.0  # ngày ngoài hạn L = 1
    assert np.array_equal(a, doi_chung.dac_trung_b2(m)) and a.shape == (224 * 224, len(doi_chung.TEN_DAC_TRUNG_B2))


def test_b2_chi_fit_tren_train():
    with pytest.raises(ValueError):
        doi_chung.B2XGB(so_cay=2).fit([_mau_b2(), _mau_b2("val")])


def test_b2_lay_pixel_co_dinh_va_bo_255():
    b = doi_chung.B2XGB(so_pixel=300)
    m = _mau_b2()
    i1, i2 = b._lay_pixel(m, np.random.default_rng(0)), b._lay_pixel(m, np.random.default_rng(0))
    assert np.array_equal(i1, i2) and len(i1) == 300 and (m["nhan"].ravel()[i1] != 255).all()


def test_b2_hoc_duoc_quy_luat_hand():
    b = doi_chung.B2XGB(so_pixel=2000, so_cay=30, thiet_bi="cpu").fit([_mau_b2(seed=s) for s in range(3)])
    m = _mau_b2(seed=9)
    p = b.predict(m)
    hl = m["nhan"] != 255
    assert p.shape == (224, 224) and ((p[hl] >= 0.5) == (m["nhan"][hl] == 1)).mean() > 0.97


def test_dac_trung_b2_thay_nan_bang_gia_tri_canh():
    m = _mau_b2()
    m["s1"][:, :5] = np.nan  # ô ngoài vùng chụp
    m["quan_trac"][:, 9] = np.nan  # GloFAS trống
    a = doi_chung.dac_trung_b2(m)
    assert not np.isnan(a).any() and (a == doi_chung.GIA_TRI_THIEU).any()
