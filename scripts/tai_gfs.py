"""Tải trước mưa dự báo GFS cho mọi chu kỳ cần dùng: chu kỳ 00Z ngày t0 + 1 = t - L + 1 với L = 1..3, cho mọi ngày
mục tiêu đã có ở data/mau/ngay_muc_tieu/*.json. Một chu kỳ phủ cả Việt Nam nên các lưu vực dùng chung cache.
Chạy: .venv/bin/python scripts/tai_gfs.py [--luong 4]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import yaml

from floodlab import du_bao
from floodlab.config import CAU_HINH_DIR, DATA_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--luong", type=int, default=4)
    luong = ap.parse_args().luong
    can = set()
    for f in sorted((DATA_DIR / "mau" / "ngay_muc_tieu").glob("*.json")):
        for n in json.loads(f.read_text())["ngay"]:
            t = date.fromisoformat(n)
            can |= {t - timedelta(days=L) + timedelta(days=1) for L in (1, 2, 3)}
    nguon_nam = yaml.safe_load((CAU_HINH_DIR / "nguon.yaml").read_text())["gfs"]["nguon_theo_nam"]
    viec = sorted(d for d in can if not (du_bao.THU_MUC_GFS / f"{d:%Y%m%d}.npy").exists())
    print(f"{len(can)} chu kỳ cần, {len(viec)} chưa có", flush=True)
    t_bd, thieu = time.time(), []

    def mot(d):
        try:
            return d, du_bao.tai_gfs(d, nguon_nam[d.year]) is not None
        except Exception as e:  # một chu kỳ lỗi không dừng cả đợt; Task 9 ghi lý do thieu_du_bao_gfs
            print(f"  {d} lỗi: {str(e)[:120]}", flush=True)
            return d, False

    with ThreadPoolExecutor(luong) as ex:
        for i, (d, ok) in enumerate(ex.map(mot, viec), 1):
            if not ok:
                thieu.append(d.isoformat())
            if i % 20 == 0:
                print(f"  {i}/{len(viec)} ({time.time() - t_bd:.0f} s)", flush=True)
    print(f"xong {len(viec)} chu kỳ, thiếu {len(thieu)}: {thieu[:20]}", flush=True)


if __name__ == "__main__":
    main()
