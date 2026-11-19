"""Danh mục trận lũ và vai trò (train/test) của từng ngày theo lưu vực."""
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from floodlab.config import CAU_HINH_DIR, CUA_SO

VaiTro = Literal["train", "test_a", "test_b"]


@dataclass(frozen=True)
class SuKien:
    ma: str
    luu_vuc: tuple[str, ...]
    bat_dau: date
    ket_thuc: date
    vai_tro: VaiTro
    ems: tuple[str, ...]
    nguon: str
    ghi_chu: str = ""


def doc_su_kien(path: Path = CAU_HINH_DIR / "su_kien.yaml") -> list[SuKien]:
    return list(_doc(str(path)))


@lru_cache
def _doc(path: str) -> tuple[SuKien, ...]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["su_kien"]
    ds = []
    for r in raw:
        lv = r["luu_vuc"]
        ds.append(SuKien(
            ma=r["ma"], luu_vuc=tuple(lv) if isinstance(lv, list) else (lv,),
            bat_dau=r["bat_dau"], ket_thuc=r["ket_thuc"], vai_tro=r["vai_tro"],
            ems=tuple(r.get("ems") or ()), nguon=r["nguon"], ghi_chu=r.get("ghi_chu", "")))
    return tuple(ds)


def _trong(ngay: date, cua_so: tuple[str, str]) -> bool:
    return date.fromisoformat(cua_so[0]) <= ngay <= date.fromisoformat(cua_so[1])


def vai_tro_cua_ngay(luu_vuc: str, ngay: date) -> str | None:
    for sk in doc_su_kien():
        if sk.vai_tro != "train" and luu_vuc in sk.luu_vuc and sk.bat_dau <= ngay <= sk.ket_thuc:
            return sk.vai_tro
    if _trong(ngay, CUA_SO["train"]):
        return "train"
    return None


# Ngưỡng trận lũ (Task 6, cố định trước khi xem test). Trung vị mùa khô thường bằng 0 nên thêm phân vị 95
# mùa khô và một sàn tuyệt đối; ngày ảnh chỉ phủ một phần nhỏ lưu vực cho tỉ lệ ngập nhiễu nên bị bỏ.
SAN_NGUONG_TRAN = 0.002
PHU_TOI_THIEU = 0.25
KHE_NGAY_TOI_DA = 30
THANG_CAY_LUA = {"hong_thai_binh": (2, 6, 7)}


def _du_phu(d):
    # Mốc phủ là trung vị chứ không phải max: vài sản phẩm thiếu lớp loại trừ có số pixel hợp lệ gấp 7 lần ngày thường.
    moc = d.pixel_hop_le[d.pixel_hop_le > 0].median()
    return d[(d.pixel_hop_le >= PHU_TOI_THIEU * moc) & d.ty_le_ngap.notna()]


def nguong_tran(d) -> float:
    """d: bảng tỉ lệ ngập theo ngày của một lưu vực (cột ngay, loai, pixel_hop_le, ty_le_ngap)."""
    kho = _du_phu(d).query("loai == 'mua_kho'").ty_le_ngap.to_numpy()
    return float(max(2 * np.median(kho), np.quantile(kho, 0.95), SAN_NGUONG_TRAN))


def tach_tran(d, nguong: float) -> list[dict]:
    """Trận = chuỗi ngày ảnh liên tiếp (đủ phủ) có tỉ lệ ngập >= nguong, hai ngày kề cách nhau tối đa 30 ngày."""
    tran, cur = [], None
    for r in _du_phu(d).sort_values("ngay").itertuples():
        ngay = r.ngay.date()
        if r.ty_le_ngap < nguong:
            cur = None
            continue
        if cur is None or (ngay - cur["ket_thuc"]).days > KHE_NGAY_TOI_DA:
            cur = {"bat_dau": ngay, "ket_thuc": ngay, "so_ngay_anh": 0, "ty_le_max": 0.0}
            tran.append(cur)
        cur["ket_thuc"] = ngay
        cur["so_ngay_anh"] += 1
        cur["ty_le_max"] = max(cur["ty_le_max"], float(r.ty_le_ngap))
    return tran


def bo_thang_cay_lua(d, luu_vuc: str):
    """Bỏ ngày train (mua_mua, mua_kho) rơi vào tháng đổ nước cấy lúa của lưu vực; ngày test giữ nguyên.
    Đồng bằng sông Hồng 2016-2023: GFM bắt nước ruộng cấy thành ngập. Tỉ lệ ngập trung vị cao nhất ở tháng 2
    (0,0054) và tháng 6-7 (0,0021; 0,0039) trong khi tháng 8-10 chỉ 0,0002-0,0007; 97-99% pixel ngập các ngày
    "trận" tháng 6-7 là ruộng (WorldCover 40). Miền Trung không có mẫu này nên không lọc."""
    thang = THANG_CAY_LUA.get(luu_vuc)
    if not thang:
        return d
    bo = d.loai.isin(["mua_mua", "mua_kho"]) & d.ngay.dt.month.isin(thang)
    return d[~bo]


def chon_ngay_muc_tieu(d, nguong: float, seed: int = 0) -> dict[date, str]:
    """Ngày mục tiêu t của một lưu vực: mọi ngày ảnh trong trận ("lu"), số ngày mùa mưa ngoài trận bằng số ngày lũ
    chọn ngẫu nhiên ("khong_lu"), và mọi ngày ảnh của cửa sổ test ("test_a"/"test_b")."""
    # Độ phủ chỉ lọc khi tìm trận (tach_tran); ngày phủ một phần vẫn là mẫu hợp lệ cho các ô được phủ.
    d = d[d.pixel_hop_le > 0]
    train = d[d.loai.isin(["mua_mua", "mua_kho"])]
    trong_tran = set()
    for t in tach_tran(train, nguong):
        trong_tran |= {n.date() for n in train.ngay if t["bat_dau"] <= n.date() <= t["ket_thuc"]}
    ngoai = sorted(n.date() for n in train.query("loai == 'mua_mua'").ngay if n.date() not in trong_tran)
    rng = np.random.default_rng(seed)
    chon = rng.choice(len(ngoai), size=min(len(trong_tran), len(ngoai)), replace=False) if ngoai else []
    out = {n: "lu" for n in trong_tran} | {ngoai[i]: "khong_lu" for i in chon}
    return out | {r.ngay.date(): r.loai for r in d.itertuples() if r.loai in ("test_a", "test_b")}
