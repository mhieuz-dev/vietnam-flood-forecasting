"""Khóa tập test (Task 6 bước 4): mọi (ô, ngày) có nhãn GFM trong cửa sổ Test A, B, kèm SHA-256 từng file nhãn,
ghi cau_hinh/tap_test_khoa.json. Chạy một lần ngay khi tải xong nhãn, trước mọi thí nghiệm, từ thư mục gốc repo.
Chạy: .venv/bin/python scripts/khoa_tap_test.py
"""
from collections import Counter

import pandas as pd

from floodlab import chia_tap
from floodlab.config import DATA_DIR, LUU_VUC, ROOT
from floodlab.luoi import luoi_o


def main():
    dong = []
    for lv in LUU_VUC:
        d = pd.read_csv(DATA_DIR / "nhan" / "_ty_le" / f"{lv}.csv", parse_dates=["ngay"])
        ngay_test = {r.ngay.date(): r.loai for r in d.itertuples() if r.loai in ("test_a", "test_b")}
        for tid in luoi_o(lv).tile_id:
            for n, vt in sorted(ngay_test.items()):
                f = DATA_DIR / "nhan" / tid / f"{n:%Y%m%d}.npy"
                if f.exists():
                    dong.append({"tile_id": tid, "ngay_t": n.isoformat(), "vai_tro": vt,
                                 "file_nhan": str(f.relative_to(ROOT))})
    tong = chia_tap.khoa_tap_test(pd.DataFrame(dong))
    print(f"{len(dong)} cặp (ô, ngày) {dict(Counter(r['vai_tro'] for r in dong))}, "
          f"{len({r['tile_id'] for r in dong})} ô, băm tổng {tong[:16]}")


if __name__ == "__main__":
    main()
