"""Tải mưa dự báo ECMWF IFS (open data, 0,25 độ) cho mọi chu kỳ 00Z mà mẫu test_a, test_b cần: t - L + 1 với L của
mẫu (Task 16, chỉ để chấm mô hình vận hành với nguồn mưa thứ hai, không train lại). Chỉ đọc ngày trong chỉ mục,
không đọc nhãn. Cache data/du_bao/ecmwf/; chạy lại thì bỏ qua chu kỳ đã có.
Chạy: .venv/bin/python scripts/tai_ecmwf.py
"""
import time
from datetime import timedelta

import pandas as pd

from floodlab import du_bao, mau
from floodlab.config import LUU_VUC


def main():
    d = pd.concat([mau.doc_chi_muc(lv) for lv in LUU_VUC], ignore_index=True)
    d = d[d.vai_tro.isin(["test_a", "test_b"])]
    can = sorted({pd.Timestamp(t).date() - timedelta(days=int(L)) + timedelta(days=1) for t, L in zip(d.ngay_t, d.L)})
    print(f"{len(d)} mẫu test, {len(can)} chu kỳ ECMWF cần ({can[0]} đến {can[-1]})", flush=True)
    t_bd, thieu = time.time(), []
    for c in can:
        if du_bao.tai_ecmwf(c) is None:
            thieu.append(c)
    print(f"xong {time.time() - t_bd:.0f} s; thiếu {len(thieu)}: {thieu[:10]}", flush=True)


if __name__ == "__main__":
    main()
