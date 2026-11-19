"""Hằng số dùng chung của dự án, lấy đúng theo thiet_ke_paper.md và plan."""
import os
from pathlib import Path

# FLOODLAB_ROOT ghi đè thư mục gốc (dữ liệu, cấu hình) khi chạy ngoài repo, vd. trên Kaggle /kaggle/input/...
ROOT = Path(os.environ.get("FLOODLAB_ROOT") or Path(__file__).resolve().parents[1])
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CAU_HINH_DIR = ROOT / "cau_hinh"

CRS = "EPSG:32648"
PIXEL_M = 20
TILE_PX = 224
TILE_M = TILE_PX * PIXEL_M  # 4480 m

LEADS = (1, 2, 3)
SEEDS = (0, 1, 2)
SO_NGAY_QUAN_TRAC = 31
NHAN_BO_QUA = 255

LUU_VUC = ("vu_gia_thu_bon", "huong", "tra_khuc", "gianh", "ca", "hong_thai_binh")
LUU_VUC_KIEM_DINH = "gianh"

# Ngày theo UTC; không ngày nào của 2024, 2025 được dùng để huấn luyện.
CUA_SO = {
    "train": ("2016-01-01", "2023-12-31"),
    "test_a": ("2024-09-01", "2024-09-30"),
    "test_b": ("2025-10-01", "2025-11-30"),
}

# (lon_min, lat_min, lon_max, lat_max): phủ lưu vực sông Hồng, Cả, Mã kể cả phần ở Trung Quốc, Lào.
KHUNG_MUA_THUONG_NGUON = (97.0, 8.0, 110.0, 26.0)
API_K = 0.90
