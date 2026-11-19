#!/bin/bash
# Tải nhãn GFM lần lượt từng lưu vực; chờ lớp tĩnh của lưu vực đó xong (cần mặt nạ nước JRC).
cd "$(dirname "$0")/.."
for lv in huong tra_khuc gianh vu_gia_thu_bon ca hong_thai_binh; do
  can=$(.venv/bin/python -c "from floodlab.luoi import luoi_o;print(len(luoi_o('$lv')))")
  while [ "$(ls data/tinh/${lv}_*.npz 2>/dev/null | wc -l)" -lt "$can" ]; do sleep 60; done
  systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python -W ignore scripts/tai_nhan.py --luu_vuc $lv
done
