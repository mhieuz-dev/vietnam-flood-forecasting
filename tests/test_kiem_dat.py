from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from floodlab import dataset, kiem_dat

P = 224
DAT = [dataset.KENH_TINH.index(k) for k in dataset.KENH_DAT]


def _lo(gia=0.0):
    return {"tinh": torch.full((1, 11, P, P), gia), "s1": torch.zeros(1, 2, P, P)}


def test_nhieu_giu_thong_ke():
    # Phép thử của Dasgupta: kênh đất + CN thay bằng nhiễu Gauss cùng trung bình, độ lệch train; trong không gian đã
    # chuẩn hóa (theo thống kê train) nghĩa là N(0, 1). Sai lệch thống kê không quá 5%, kênh khác giữ nguyên
    f = kiem_dat.nhieu(seed=0)
    b = f(_lo(3.0), SimpleNamespace(tile_id="a"))
    x = b["tinh"][0, DAT]
    assert abs(float(x.mean())) < 0.05 and abs(float(x.std()) - 1) < 0.05
    khac = [i for i in range(11) if i not in DAT]
    assert torch.all(b["tinh"][0, khac] == 3.0)
    b2 = kiem_dat.nhieu(seed=0)(_lo(3.0), SimpleNamespace(tile_id="a"))
    assert torch.equal(b2["tinh"], b["tinh"])  # tái lập theo seed


def test_hoan_vi_giu_tap_gia_tri(monkeypatch):
    # Hoán vị bản đồ đất giữa các ô trong cùng vai trò: song ánh theo ô nên tập giá trị trên cả vai trò giữ nguyên
    o = [f"t{i}" for i in range(6)]
    chi_muc = pd.DataFrame({"tile_id": o + o, "luu_vuc": "huong", "i_o": list(range(6)) * 2})
    tinh_tho = {i: np.full((12, P, P), float(i), np.float32) for i in range(6)}
    monkeypatch.setattr(kiem_dat.mau, "mo_zarr", lambda lv: {"tinh": tinh_tho})
    monkeypatch.setattr(kiem_dat.dataset, "_thong_ke_chuan_hoa",
                        lambda: (np.zeros((11, 1, 1), np.float32), np.ones((11, 1, 1), np.float32), None, None))
    f = kiem_dat.hoan_vi(chi_muc, seed=0)
    ra = [float(f(_lo(-1.0), SimpleNamespace(tile_id=t))["tinh"][0, DAT[0], 0, 0]) for t in o]
    assert sorted(ra) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] and ra != [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    b = f(_lo(-1.0), SimpleNamespace(tile_id="t0"))
    assert torch.all(b["tinh"][0, [i for i in range(11) if i not in DAT]] == -1.0)  # chỉ đổi kênh đất
