"""Tải ảnh S2 mùa khô (trung vị tháng 1-4, Task 15) cho đối chứng B4: mỗi cặp (ô, năm của ngày t) có trong chỉ mục
mẫu, mọi vai trò (train, val, test) vì B4 chấm trên test ở Task 16. Ảnh là bối cảnh mặt đất trước mùa lũ cùng năm
(pixel trống bù từ mùa khô năm trước, xem s2._anh),
không chứa thông tin ngày t. Chạy lại được: cặp đã có file thì bỏ qua.
Chạy: .venv/bin/python scripts/tai_s2.py [--luu_vuc huong] [--chi_dem]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import numpy as np
import pandas as pd

from floodlab import dataset, mau, s2
from floodlab.config import EE_PROJECT, RESULTS_DIR
from floodlab.luoi import luoi_o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luu_vuc", nargs="*", default=list(dataset.LUU_VUC))
    ap.add_argument("--chi_dem", action="store_true")
    a = ap.parse_args()
    ee.Initialize(project=EE_PROJECT)
    for lv in a.luu_vuc:
        d = mau.doc_chi_muc(lv)
        cap = sorted({(t, n) for t, n in zip(d.tile_id, pd.to_datetime(d.ngay_t).dt.year)})
        hinh = dict(zip(*luoi_o(lv)[["tile_id", "geometry"]].T.values))
        da_co = sum(s2._duong(t, n).exists() for t, n in cap)
        print(f"{lv}: {len(cap)} cặp ô-năm, đã có {da_co}", flush=True)
        if a.chi_dem:
            continue
        t_bd, xong, tk = time.time(), 0, {}
        with ThreadPoolExecutor(6) as ex:
            for (t, n), m in zip(cap, ex.map(lambda v: s2.anh_mua_kho(v[0], hinh[v[0]], int(v[1])), cap)):
                xong += 1
                tk.setdefault(str(n), []).append((float(np.isnan(m).all(axis=0).mean()), s2.ty_le_bu(t, n)))
                if xong % 500 == 0:
                    print(f"  {xong}/{len(cap)} ({time.time() - t_bd:.0f} s)", flush=True)
        f = RESULTS_DIR / "s2_thong_ke.json"
        cu = json.loads(f.read_text()) if f.exists() else {}
        cu[lv] = {n: {"so_o": len(v), "ty_le_nan_tb": round(float(np.mean([a for a, _ in v])), 4),
                      "so_o_qua_nua_nan": sum(a > 0.5 for a, _ in v), "ty_le_bu_tb": round(float(np.mean([b for _, b in v])), 4)}
                  for n, v in sorted(tk.items())}
        f.write_text(json.dumps(cu, indent=1, ensure_ascii=False))
        print(f"{lv} xong: {xong} cặp, {time.time() - t_bd:.0f} s; {json.dumps(cu[lv])}", flush=True)


if __name__ == "__main__":
    main()
