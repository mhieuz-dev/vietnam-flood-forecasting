"""Hàm dự đoán cho danh_gia (Task 16): nạp checkpoints/{ten}/best.pt, chuẩn hóa mẫu thô như lúc train
(dataset.mau_cho_mo_hinh), trả xác suất đã hiệu chỉnh nhiệt độ (sigmoid(z / T), T lưu trong checkpoint).
bien_doi(lô, chỉ mục) cho phép thử đất (kiem_dat) đổi đầu vào đã chuẩn hóa trước khi vào mô hình."""
from types import SimpleNamespace

import numpy as np
import torch

from floodlab import dataset, huan_luyen

DIR_CKPT = huan_luyen.DIR_CKPT


def tai(ten: str, thiet_bi: str | None = None):
    tb = torch.device(thiet_bi or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(DIR_CKPT / ten / "best.pt", map_location="cpu", weights_only=False)
    m = huan_luyen.tao_mo_hinh(ck["cfg"] | {"pretrained": False})  # trọng số lấy từ checkpoint, không tải lại
    m.load_state_dict(ck["mo_hinh"])
    return m.eval().to(tb), ck, tb


def ham_du_doan(ten: str, thiet_bi: str | None = None, bien_doi=None):
    m, ck, tb = tai(ten, thiet_bi)
    cfg, T = ck["cfg"], float(ck["nhiet_do"])
    bo, co_s2 = set(cfg.get("bo") or []), cfg["mo_hinh"] == "prithvi_b4"

    def f(mau_tho: dict) -> np.ndarray:
        r = SimpleNamespace(**mau_tho["chi_muc"])
        b = dataset.mau_cho_mo_hinh(mau_tho, r, bo=bo, co_s2=co_s2)
        if bien_doi is not None:
            b = bien_doi(b, r)
        b = {k: v.to(tb) for k, v in b.items()}
        with torch.no_grad(), torch.autocast(tb.type, dtype=torch.float16, enabled=tb.type == "cuda"):
            z = m(b)[0, 0]
        return torch.sigmoid(z.float() / T).cpu().numpy().astype(np.float32)

    return f
