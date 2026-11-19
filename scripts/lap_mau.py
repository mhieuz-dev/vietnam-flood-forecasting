"""Lắp mẫu và ghi shard zarr cho một hoặc mọi lưu vực (Task 9), in số mẫu, lý do loại và dung lượng trên đĩa.
Chạy: .venv/bin/python scripts/lap_mau.py --luu_vuc huong   (bỏ --luu_vuc để chạy mọi lưu vực)
"""
import argparse
import time

from floodlab import mau
from floodlab.config import LUU_VUC


def _dung_luong(p) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luu_vuc", choices=LUU_VUC)
    lv_chay = [ap.parse_args().luu_vuc] if ap.parse_args().luu_vuc else list(LUU_VUC)
    for lv in lv_chay:
        t0 = time.time()
        dem = mau.lap(lv)
        gb = _dung_luong(mau.DIR_MAU / f"{lv}.zarr") / 1e9
        print(f"{lv}: {dem}, {gb:.2f} GB, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
