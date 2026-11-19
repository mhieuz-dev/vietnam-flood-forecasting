"""Phép thử đất trên test (Task 16, RQ2): mô hình có thật sự dùng thông tin đất không.

Cả hai trả hàm bien_doi(lô đã chuẩn hóa, chỉ mục) cắm vào du_doan.ham_du_doan, đổi các kênh đất và CN (dataset.KENH_DAT)
trước khi vào mô hình; mọi kênh khác giữ nguyên.
- nhieu: thay bằng nhiễu Gauss cùng trung bình, độ lệch train (phép thử của Dasgupta và cs., arXiv:2208.09135); trong
  không gian đã chuẩn hóa theo thống kê train đó là N(0, 1).
- hoan_vi: hoán vị bản đồ đất giữa các ô trong cùng vai trò (song ánh theo ô, nên tập giá trị trên cả vai trò giữ nguyên
  và tương quan không gian trong một ô vẫn còn); lớp tĩnh của ô kia đọc từ zarr mẫu, đúng dạng mô hình đã thấy.
"""
import numpy as np
import pandas as pd
import torch

from floodlab import dataset, mau

KENH = [dataset.KENH_TINH.index(k) for k in dataset.KENH_DAT]


def nhieu(seed: int):
    g = torch.Generator().manual_seed(seed)

    def f(b: dict, r) -> dict:
        b = dict(b)
        t = b["tinh"].clone()
        t[:, KENH] = torch.randn(t[:, KENH].shape, generator=g)
        b["tinh"] = t
        return b

    return f


def hoan_vi(chi_muc: pd.DataFrame, seed: int):
    o = chi_muc.drop_duplicates("tile_id")[["tile_id", "luu_vuc", "i_o"]].reset_index(drop=True)
    p = np.random.default_rng(seed).permutation(len(o))
    sang = {o.tile_id[i]: (o.luu_vuc[j], int(o.i_o[j])) for i, j in enumerate(p)}
    tb, ls, _, _ = dataset._thong_ke_chuan_hoa()
    zarr_lv = {}

    def f(b: dict, r) -> dict:
        lv, i_o = sang[r.tile_id]
        if lv not in zarr_lv:
            zarr_lv.clear()
            zarr_lv[lv] = mau.mo_zarr(lv)
        tinh = (np.asarray(zarr_lv[lv]["tinh"][i_o], np.float32)[1:] - tb) / ls
        b = dict(b)
        t = b["tinh"].clone()
        t[0, KENH] = torch.from_numpy(np.nan_to_num(tinh[KENH]))
        b["tinh"] = t
        return b

    return f
