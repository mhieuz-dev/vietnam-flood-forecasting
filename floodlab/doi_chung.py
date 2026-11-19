"""Đối chứng không học sâu: B0 tần suất ngập lịch sử, B1 ngưỡng HAND và mưa, B6 giữ trạng thái (B2 XGBoost ở Task 11 bước 3).

Ngưỡng của B1 chỉ chọn trên mẫu val (lưu vực gianh); truyền mẫu khác vào là lỗi.
"""
from collections import Counter
from datetime import date

import numpy as np

from floodlab import chi_so
from floodlab.config import CUA_SO, DATA_DIR, NHAN_BO_QUA

DIR_NHAN = DATA_DIR / "nhan"
DIR_B0 = DATA_DIR / "doi_chung" / "b0"
LUOI_HAND = (1, 2, 3, 5, 7, 10, 15, 20, 30)
PHAN_VI_API = (50, 60, 70, 80, 90, 95, 99)


def b6_giu_trang_thai(nhan_trang_thai: np.ndarray) -> np.ndarray:
    """Dự báo = nhãn ngập mới nhất đã biết ở t0; pixel không có nhãn (255) coi là không ngập."""
    return np.where(nhan_trang_thai == 1, 1.0, 0.0)


def b0_tan_suat(tile_id: str, dung_cache: bool = True) -> np.ndarray:
    """Tần suất ngập GFM của từng pixel trên các ngày huấn luyện; pixel 255 không vào mẫu số. Lưu cache ở DIR_B0
    vì nhãn thô (16 GB) được đưa lên Drive sau khi lắp mẫu."""
    p = DIR_B0 / f"{tile_id}.npy"
    if dung_cache and p.exists():
        return np.load(p)
    tu, den = (date.fromisoformat(x) for x in CUA_SO["train"])
    ngap = so = None
    for f in sorted((DIR_NHAN / tile_id).glob("*.npy")):
        s = f.stem
        if not tu <= date(int(s[:4]), int(s[4:6]), int(s[6:8])) <= den:
            continue
        a = np.load(f)
        if ngap is None:
            ngap, so = np.zeros(a.shape, np.float32), np.zeros(a.shape, np.float32)
        hop_le = a != NHAN_BO_QUA
        ngap += (a == 1) & hop_le
        so += hop_le
    if ngap is None:
        raise FileNotFoundError(f"không có nhãn train cho {tile_id} ở {DIR_NHAN}")
    b0 = np.divide(ngap, so, out=np.zeros_like(ngap), where=so > 0)
    if dung_cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, b0)
    return b0


def b1_hand_mua(hand: np.ndarray, api: np.ndarray, nguong_hand: float, nguong_api: float) -> np.ndarray:
    """Ngập nếu pixel thấp gần sông (HAND <= ngưỡng) và mưa tích lũy trước đủ lớn (API >= ngưỡng)."""
    return ((hand <= nguong_hand) & (api >= nguong_api)).astype(np.float32)


def chon_nguong_b1(mau: list[dict]) -> tuple[float, float]:
    """Lưới HAND x phân vị API, chọn cặp có CSI gộp lớn nhất. mau: dict có vai_tro, hand, api, nhan."""
    vai_tro = Counter(m["vai_tro"] for m in mau)
    if set(vai_tro) != {"val"}:
        raise ValueError(f"ngưỡng B1 chỉ được chọn trên val, nhận {dict(vai_tro)}")
    hand = np.concatenate([np.ravel(m["hand"]) for m in mau])
    api = np.concatenate([np.ravel(np.broadcast_to(m["api"], np.shape(m["hand"]))) for m in mau])
    nhan = np.concatenate([np.ravel(m["nhan"]) for m in mau])
    tot, csi_tot = None, -1.0
    for nh in LUOI_HAND:
        for na in np.unique(np.percentile(api, PHAN_VI_API)):
            csi = chi_so.tu_bang(**chi_so.bang_nham_lan(b1_hand_mua(hand, api, nh, na), nhan))["csi"]
            if csi > csi_tot:
                tot, csi_tot = (float(nh), float(na)), csi
    return tot


def _ten_dac_trung():
    from floodlab.mau import KENH_TINH

    return (*KENH_TINH, "s1_vv", "s1_vh", "s1_vv_tru_vh",
            "mua_o_1n", "mua_o_3n", "mua_o_7n", "mua_o_30n", "mua_tn_3n", "mua_tn_7n", "mua_tn_30n", "api_t0",
            "swvl1_t0", "swvl2_t0", "swvl3_t0", "swvl4_t0", "t2m_7n", "rh_7n", "glofas_t0", "glofas_max_7n",
            "du_bao_mua_o", "du_bao_mua_tn", "du_bao_mua_o_max", "L")


TEN_DAC_TRUNG_B2 = _ten_dac_trung()
GIA_TRI_THIEU = -9999.0


def dac_trung_b2(m: dict) -> np.ndarray:
    """(224*224, F) theo TEN_DAC_TRUNG_B2: lớp tĩnh, S1 trạng thái, quan trắc 31 ngày đến t0 gộp lại (cột theo
    khi_tuong.COT), mưa dự báo trong hạn L (nhân mặt nạ nên ngày ngoài hạn không lọt vào) và L. NaN (S1 ngoài vùng
    chụp, GloFAS trống) thay bằng GIA_TRI_THIEU: có NaN thì QuantileDMatrix lưu dạng thưa, tốn thêm 1,01 GB cho
    0,30 GB dữ liệu so với 0,28 GB khi đã thay (đo 29/10); cây vẫn tách được giá trị này thành nhánh riêng."""
    qt, mn = m["quan_trac"], m["mat_na"]
    db = m["du_bao"] * mn[:, None]
    vo = np.nansum
    vo_huong = [vo(qt[-1:, 0]), vo(qt[-3:, 0]), vo(qt[-7:, 0]), vo(qt[-30:, 0]), vo(qt[-3:, 1]), vo(qt[-7:, 1]),
                vo(qt[-30:, 1]), qt[-1, 8], *qt[-1, 4:8], np.nanmean(qt[-7:, 2]), np.nanmean(qt[-7:, 3]), qt[-1, 9],
                np.nanmax(qt[-7:, 9]) if np.isfinite(qt[-7:, 9]).any() else np.nan,
                db[:, 0].sum(), db[:, 1].sum(), db[:, 0].max(), float(mn.sum())]
    s1 = m["s1"]
    anh = np.concatenate([m["tinh"], s1, s1[:1] - s1[1:2]]).reshape(len(m["tinh"]) + 3, -1).T
    x = np.hstack([anh, np.broadcast_to(np.asarray(vo_huong, np.float32), (anh.shape[0], len(vo_huong)))])
    return np.nan_to_num(x.astype(np.float32), nan=GIA_TRI_THIEU)


class B2XGB:
    """XGBoost theo pixel trên TEN_DAC_TRUNG_B2. fit nhận các mẫu (dict của mau.doc_mau kèm khóa vai_tro), chỉ
    nhận train; mỗi mẫu lấy ngẫu nhiên tối đa so_pixel pixel có nhãn (khác 255) bằng một bộ sinh seed cố định.
    Dữ liệu đưa vào XGBoost theo lô qua DataIter, không nối thành một mảng: 8.000 mẫu x 2.000 pixel x 38 đặc trưng
    là 2,4 GB float32, nối lại rồi để XGBoost chép thêm thì vượt 10 GB (bị OOM killer giết 29/10)."""

    LO = 200  # số mẫu mỗi lô

    def __init__(self, so_pixel: int = 2000, seed: int = 0, so_cay: int = 400, thiet_bi: str = "cuda", **tham_so):
        self.so_pixel, self.seed, self.so_cay = so_pixel, seed, so_cay
        self.tham_so = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 8, "eta": 0.1,
                        "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "device": thiet_bi,
                        "seed": seed} | tham_so
        self.mo_hinh = None

    def _lay_pixel(self, m: dict, rng) -> np.ndarray:
        hl = np.flatnonzero(m["nhan"].ravel() != NHAN_BO_QUA)
        return np.sort(rng.choice(hl, size=min(self.so_pixel, len(hl)), replace=False))

    def _lo(self, cac_mau, vai_tro: str):
        """Danh sách lô (X, y), mỗi lô gom LO mẫu."""
        rng = np.random.default_rng(self.seed)
        lo, X, y = [], [], []
        for m in cac_mau:
            if m.get("vai_tro") != vai_tro:
                raise ValueError(f"B2 chỉ nhận mẫu {vai_tro} ở bước này, gặp {m.get('vai_tro')}")
            i = self._lay_pixel(m, rng)
            X.append(dac_trung_b2(m)[i])
            y.append((m["nhan"].ravel()[i] == 1).astype(np.float32))
            if len(X) == self.LO:
                lo.append((np.concatenate(X), np.concatenate(y)))
                X, y = [], []
        if X:
            lo.append((np.concatenate(X), np.concatenate(y)))
        return lo

    @staticmethod
    def _ma_tran(lo, ref=None):
        import xgboost as xgb

        class _Lap(xgb.DataIter):
            def __init__(self):
                self.i = 0
                super().__init__()

            def next(self, input_data):
                if self.i == len(lo):
                    return False
                input_data(data=lo[self.i][0], label=lo[self.i][1])
                self.i += 1
                return True

            def reset(self):
                self.i = 0

        return xgb.QuantileDMatrix(_Lap(), ref=ref, max_bin=256)

    def fit(self, cac_mau, cac_mau_val=None):
        """cac_mau_val (lưu vực kiểm định) chỉ dùng dừng sớm; số cây tốt nhất ở self.mo_hinh.best_iteration."""
        import xgboost as xgb

        lo = self._lo(cac_mau, "train")
        self.so_dong_train = int(sum(len(b[1]) for b in lo))
        self.ty_le_ngap_train = float(sum(b[1].sum() for b in lo) / self.so_dong_train)
        dt = self._ma_tran(lo)
        del lo
        dg, dung = [], None
        if cac_mau_val is not None:
            dg, dung = [(self._ma_tran(self._lo(cac_mau_val, "val"), ref=dt), "val")], 30
        self.mo_hinh = xgb.train(self.tham_so, dt, num_boost_round=self.so_cay, evals=dg,
                                 early_stopping_rounds=dung, verbose_eval=False)
        return self

    def predict(self, m: dict) -> np.ndarray:
        it = (0, self.mo_hinh.best_iteration + 1) if self.mo_hinh.attr("best_iteration") is not None else (0, 0)
        p = self.mo_hinh.inplace_predict(dac_trung_b2(m), iteration_range=it)
        return p.reshape(m["nhan"].shape).astype(np.float32)
