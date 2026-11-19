"""Chạy một lượt huấn luyện trong cau_hinh/thi_nghiem.yaml theo tên; khóa ghi đè được truyền dạng khoa=gia_tri.
Chạy: .venv/bin/python scripts/huan_luyen.py hop_nhat_gfs_s0 [so_epoch=2 batch=12]
"""
import sys

import yaml

from floodlab import huan_luyen
from floodlab.config import CAU_HINH_DIR


def main():
    ten, ghi_de = sys.argv[1], sys.argv[2:]
    cfg = next(l for l in yaml.safe_load((CAU_HINH_DIR / "thi_nghiem.yaml").read_text())["luot"] if l["ten"] == ten)
    for kv in ghi_de:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    print(huan_luyen.chay(cfg))


if __name__ == "__main__":
    main()
