import json
from datetime import date

import numpy as np
import pandas as pd
import torch

from floodlab import chi_so, dataset, huan_luyen

P = 224


class _TapGia(torch.utils.data.Dataset):
    """8 mẫu; ngập khi kênh VV > 0, có một dải nhãn 255."""

    def __init__(self, n=8):
        g = torch.Generator().manual_seed(0)
        self.m = []
        for i in range(n):
            s1 = torch.randn(2, P, P, generator=g)
            s1[0] = torch.nn.functional.avg_pool2d(s1[:1], 15, 1, 7)[0] * 4
            nhan = (s1[0] > 0).long()
            nhan[:8] = 255
            self.m.append({"s1": s1, "dem": torch.zeros(1, P, P), "tinh": torch.zeros(11, P, P),
                           "quan_trac": torch.zeros(31, 10), "du_bao": torch.zeros(3, 2),
                           "mat_na_du_bao": torch.tensor([True, False, False]), "L": 1, "khoang_cach_s1": 0.1,
                           "nhan": nhan, "mau_id": f"m{i}", "bo": ""})

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        return self.m[i]


def test_hoc_thuoc_8_mau(monkeypatch, tmp_path):
    monkeypatch.setattr(huan_luyen, "DIR_CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(huan_luyen, "DIR_KQ", tmp_path / "kq")
    tap = _TapGia()
    cfg = {"ten": "thu", "mo_hinh": "unet_b3", "bo": [], "seed": 0, "so_epoch": 50, "so_mau_epoch": 8, "batch": 8,
           "lr": 1e-3, "lr_lung": 1e-4, "kien_nhan": 100, "so_luong": 0}
    j = json.loads(huan_luyen.chay(cfg, tap_train=tap, tap_val=tap).read_text())
    dc = j["duong_cong"]
    # CSI train tính trên batch lúc học; CSI val ở chế độ eval dao động vì BatchNorm chưa ổn định với 8 mẫu
    assert dc[-1]["loss_train"] <= 0.5 * dc[0]["loss_train"] and max(e["csi_train"] for e in dc[-5:]) > 0.9
    assert (tmp_path / "ckpt" / "thu" / "best.pt").exists() and j["nhiet_do"] > 0


def test_mat_mat_bo_qua_255():
    z = torch.randn(2, P, P)
    y = (torch.rand(2, P, P) > 0.5).long()
    y2 = y.clone()
    y2[:, :10] = 255
    z2 = z.clone()
    z2[:, :10] = 1000.0  # logit vô lý ở vùng 255 không được ảnh hưởng
    a = huan_luyen.mat_mat(z2, y2)
    b = huan_luyen.mat_mat(z, y2)
    assert torch.isfinite(a) and torch.allclose(a, b)


def test_nhiet_do_giam_ece():
    g = torch.Generator().manual_seed(0)
    goc = torch.randn(200_000, generator=g) * 1.5
    nhan = (torch.rand(200_000, generator=g) < torch.sigmoid(goc)).long()
    qua_tu_tin = goc * 4
    T = huan_luyen.nhiet_do(qua_tu_tin, nhan)
    ece = lambda z: chi_so.tinh(torch.sigmoid(z).numpy(), nhan.numpy())["ece_15"]
    assert 3 < T < 5 and ece(qua_tu_tin / T) < 0.5 * ece(qua_tu_tin)


def test_lobo_khong_co_luu_vuc_bo_ra_trong_train(monkeypatch, tmp_path):
    d = pd.DataFrame({"mau_id": ["a", "b", "c"], "luu_vuc": ["huong", "ca", "gianh"], "vai_tro": ["train", "train", "val"],
                      "tile_id": ["huong_1_1", "ca_1_1", "gianh_1_1"], "ngay_t": date(2020, 10, 1), "L": 1,
                      "khoang_cach_s1_ngay": 3, "ty_le_ngap": 0.1, "i_o": 0, "i_s1": 0, "i_nhan": 0, "i_mau": 0})
    monkeypatch.setattr(dataset, "_chi_muc", lambda vt: d[d.vai_tro == vt].reset_index(drop=True))
    f = tmp_path / "ch.json"
    f.write_text(json.dumps({"tinh_tb": [0] * 11, "tinh_ls": [1] * 11, "qt_tb": [0] * 10, "qt_ls": [1] * 10}))
    monkeypatch.setattr(dataset, "F_CHUAN_HOA", f)
    tr, va = huan_luyen.tao_tap({"nguon_du_bao_train": "gfs", "bo": [], "luu_vuc_bo_ra": "huong"})
    assert set(tr.d.luu_vuc) == {"ca"} and set(va.d.luu_vuc) == {"gianh"}
    tr, _ = huan_luyen.tao_tap({"nguon_du_bao_train": "gfs", "bo": [], "luu_vuc_bo_ra": None})
    assert set(tr.d.luu_vuc) == {"huong", "ca"}


def test_lich_lr_cosine_giam_dan(monkeypatch, tmp_path):
    monkeypatch.setattr(huan_luyen, "DIR_CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(huan_luyen, "DIR_KQ", tmp_path / "kq")
    cfg = {"ten": "thu", "mo_hinh": "unet_b3", "bo": [], "seed": 0, "so_epoch": 4, "so_mau_epoch": 8, "batch": 8,
           "lr": 1e-3, "lr_lung": 1e-4, "kien_nhan": 100, "so_luong": 0}
    tap = _TapGia()
    co_dinh = json.loads(huan_luyen.chay(cfg, tap_train=tap, tap_val=tap).read_text())["duong_cong"]
    cos = json.loads(huan_luyen.chay(cfg | {"lich_lr": "cosine"}, tap_train=tap, tap_val=tap).read_text())["duong_cong"]
    assert [e["lr"] for e in co_dinh] == [1e-3] * 4
    lr = [e["lr"] for e in cos]
    assert lr[0] == 1e-3 and all(a > b for a, b in zip(lr, lr[1:])) and lr[-1] < 0.2e-3


def test_b4_dung_prithvi_va_tap_co_s2(monkeypatch):
    from floodlab.mo_hinh.prithvi_b4 import MoHinhB4
    goi = []
    monkeypatch.setattr(huan_luyen.dataset, "MauNgap", lambda *a, **k: goi.append(k) or k)
    cfg = {"mo_hinh": "prithvi_b4", "pretrained": False, "nho_de_test": True, "bo": [], "nguon_du_bao_train": "gfs"}
    assert isinstance(huan_luyen.tao_mo_hinh(cfg), MoHinhB4)
    huan_luyen.tao_tap(cfg)
    assert [k["co_s2"] for k in goi] == [True, True]
    goi.clear()
    huan_luyen.tao_tap(cfg | {"mo_hinh": "hop_nhat"})
    assert [k["co_s2"] for k in goi] == [False, False]


def test_b5_dung_utae_gff_khong_can_s2(monkeypatch):
    from floodlab.mo_hinh.utae_b5 import MoHinhB5
    goi = []
    monkeypatch.setattr(huan_luyen.dataset, "MauNgap", lambda *a, **k: goi.append(k) or k)
    cfg = {"mo_hinh": "utae_b5", "bo": [], "nguon_du_bao_train": "gfs"}
    assert isinstance(huan_luyen.tao_mo_hinh(cfg), MoHinhB5)
    huan_luyen.tao_tap(cfg)
    assert [k["co_s2"] for k in goi] == [False, False]


def test_tich_luy_gradient_giu_batch_hieu_dung(monkeypatch, tmp_path):
    # Prithvi B4 batch 16 không vừa 8 GB kể cả checkpoint (trạng thái AdamW 300M ~2,4 GB, 05/11): tich_luy=2 chạy
    # micro-batch batch/2, cập nhật sau mỗi 2 micro-batch, nên batch hiệu dụng, lr, số mẫu mỗi epoch giữ nguyên
    monkeypatch.setattr(huan_luyen, "DIR_CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(huan_luyen, "DIR_KQ", tmp_path / "kq")
    buoc, co_lo = [], []
    goc_step = torch.optim.AdamW.step
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **k: buoc.append(1) or goc_step(self, *a, **k))
    goc_mat = huan_luyen.mat_mat
    monkeypatch.setattr(huan_luyen, "mat_mat", lambda z, y: co_lo.append(len(y)) or goc_mat(z, y))
    cfg = {"ten": "thu", "mo_hinh": "unet_b3", "bo": [], "seed": 0, "so_epoch": 1, "so_mau_epoch": 16, "batch": 8,
           "tich_luy": 2, "lr": 1e-3, "lr_lung": 1e-4, "kien_nhan": 100, "so_luong": 0}
    huan_luyen.chay(cfg, tap_train=_TapGia(), tap_val=_TapGia())
    assert co_lo[:4] == [4, 4, 4, 4] and len(buoc) == 2
