"""Vòng huấn luyện mô hình học sâu (Task 13): AdamW, fp16 trên GPU, loss BCE + Dice bỏ qua nhãn 255, dừng sớm theo
CSI val (ngưỡng 0,5), rồi hiệu chỉnh nhiệt độ trên val. Ghi checkpoints/{ten}/best.pt và results/huan_luyen/{ten}.json.

Một "epoch" là so_mau_epoch mẫu rút từ lay_mau_can_bang (50% có ngập), seed đổi theo epoch; mặc định bằng số mẫu
train. LOBO: luu_vuc_bo_ra bị bỏ khỏi train, val vẫn là gianh (siêu tham số giữ nguyên, spec §8).
"""
import json
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler

from floodlab import chia_tap, dataset
from floodlab.config import NHAN_BO_QUA, RESULTS_DIR, ROOT

DIR_CKPT = ROOT / "checkpoints"
DIR_KQ = RESULTS_DIR / "huan_luyen"
SO_PIXEL_NHIET_DO = 2_000_000


def tao_tap(cfg: dict):
    kw = {"nguon_du_bao": cfg["nguon_du_bao_train"], "bo": set(cfg.get("bo") or []),
          "co_s2": cfg.get("mo_hinh") == "prithvi_b4"}
    return (dataset.MauNgap("train", tang_cuong=True, bo_luu_vuc=cfg.get("luu_vuc_bo_ra"), **kw),
            dataset.MauNgap("val", **kw))


def tao_mo_hinh(cfg: dict) -> torch.nn.Module:
    bo = set(cfg.get("bo") or [])
    if cfg["mo_hinh"] == "hop_nhat":
        from floodlab.mo_hinh.hop_nhat import MoHinhHopNhat

        return MoHinhHopNhat(bo_ma_hoa=cfg.get("bo_ma_hoa", "terramind_v1_base"), pretrained=cfg.get("pretrained", True),
                             bo=bo)
    if cfg["mo_hinh"] == "unet_b3":
        from floodlab.mo_hinh.unet import UNetB3

        return UNetB3(bo=bo)
    if cfg["mo_hinh"] == "prithvi_b4":
        from floodlab.mo_hinh.prithvi_b4 import MoHinhB4

        return MoHinhB4(pretrained=cfg.get("pretrained", True), nho_de_test=cfg.get("nho_de_test", False), bo=bo,
                        ckpt_lung=cfg.get("ckpt_lung", False))
    if cfg["mo_hinh"] == "utae_b5":
        from floodlab.mo_hinh.utae_b5 import MoHinhB5

        return MoHinhB5(bo=bo)
    raise NotImplementedError(cfg["mo_hinh"])


def mat_mat(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BCE trung bình trên pixel có nhãn + (1 - Dice) gộp cả batch; pixel 255 không đóng góp."""
    hl = (y != NHAN_BO_QUA).float()
    t = (y == 1).float()
    z = torch.where(hl > 0, z.float(), torch.zeros_like(z, dtype=torch.float32))
    bce = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * hl).sum() / hl.sum().clamp(min=1)
    p = torch.sigmoid(z) * hl
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


def nhiet_do(logits: torch.Tensor, nhan: torch.Tensor) -> float:
    """Nhiệt độ T > 0 cực tiểu BCE của logits / T trên pixel có nhãn (nhãn 255 bị bỏ)."""
    hl = nhan != NHAN_BO_QUA
    z, t = logits[hl].double(), (nhan[hl] == 1).double()
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def buoc():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(z / log_t.exp(), t)
        l.backward()
        return l

    opt.step(buoc)
    return float(log_t.exp())


def _len_thiet_bi(b, tb):
    """Chỉ giữ tensor: DataParallel không chia được danh sách chuỗi (mau_id, bo)."""
    return {k: v.to(tb, non_blocking=True) for k, v in b.items() if torch.is_tensor(v)}


def _danh_gia_val(m, dl, tb, lay_logit=False, rng=None):
    tp = fp = fn = 0
    zs, ys = [], []
    m.eval()
    with torch.no_grad():
        for b in dl:
            b = _len_thiet_bi(b, tb)
            with torch.autocast(tb.type, dtype=torch.float16, enabled=tb.type == "cuda"):
                z = m(b)[:, 0].float()
            y = b["nhan"]
            hl = y != NHAN_BO_QUA
            du, that = (z >= 0) & hl, (y == 1)
            tp += int((du & that).sum())
            fp += int((du & ~that & hl).sum())
            fn += int((~du & that & hl).sum())
            if lay_logit:
                zh, yh = z[hl].cpu(), y[hl].cpu()
                k = min(len(zh), (SO_PIXEL_NHIET_DO // max(len(dl.dataset), 1) + 1) * len(y))  # theo mẫu, không theo batch
                i = torch.from_numpy(rng.choice(len(zh), k, replace=False)) if len(zh) else torch.zeros(0, dtype=torch.long)
                zs.append(zh[i])
                ys.append(yh[i])
    csi = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return (csi, torch.cat(zs), torch.cat(ys)) if lay_logit else csi


def chay(cfg: dict, tap_train=None, tap_val=None):
    t_bd = time.time()
    torch.manual_seed(cfg["seed"])
    tb = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tap_train is None:
        tap_train, tap_val = tao_tap(cfg)
    m = tao_mo_hinh(cfg).to(tb)
    # Kaggle 2 x T4: chia batch cho mọi GPU; checkpoint vẫn lưu từ m (không có tiền tố module.)
    mh = torch.nn.DataParallel(m) if tb.type == "cuda" and torch.cuda.device_count() > 1 else m
    lung = list(getattr(m, "lung", torch.nn.Module()).parameters())
    id_lung = {id(p) for p in lung}
    nhom = [{"params": [p for p in m.parameters() if id(p) not in id_lung], "lr": cfg["lr"]}]
    if lung:
        nhom.append({"params": lung, "lr": cfg["lr_lung"]})
    opt = torch.optim.AdamW(nhom, weight_decay=cfg.get("weight_decay", 0.01))
    scaler = torch.amp.GradScaler(enabled=tb.type == "cuda")
    # lich_lr=cosine: giảm theo epoch về 0 ở so_epoch; mặc định giữ lr cố định
    lich = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["so_epoch"]) if cfg.get("lich_lr") == "cosine" else None
    so_mau = cfg.get("so_mau_epoch") or len(tap_train)
    # tich_luy: micro-batch batch/tich_luy, cập nhật sau mỗi tich_luy micro-batch (batch hiệu dụng giữ nguyên)
    k_tl = int(cfg.get("tich_luy", 1))
    kw = {"batch_size": cfg["batch"] // k_tl, "num_workers": cfg.get("so_luong", 4), "pin_memory": tb.type == "cuda"}
    dl_val = DataLoader(tap_val, shuffle=False, **kw)
    thu_muc = DIR_CKPT / cfg["ten"]
    thu_muc.mkdir(parents=True, exist_ok=True)
    dc, tot, khong_tien, epoch_tot = [], -1.0, 0, -1
    for ep in range(cfg["so_epoch"]):
        t_ep = time.time()
        if isinstance(tap_train, dataset.MauNgap):
            lay = dataset.lay_mau_can_bang(tap_train, seed=cfg["seed"] * 1000 + ep, so_mau=so_mau,
                                           can_luu_vuc=cfg.get("can_luu_vuc", False))
        else:
            lay = RandomSampler(tap_train, replacement=True, num_samples=so_mau,
                                generator=torch.Generator().manual_seed(cfg["seed"] * 1000 + ep))
        m.train()
        tong, n, dem = 0.0, 0, np.zeros(3)
        dl_train = DataLoader(tap_train, sampler=lay, drop_last=False, **kw)
        opt.zero_grad(set_to_none=True)
        for i_b, b in enumerate(dl_train):
            b = _len_thiet_bi(b, tb)
            with torch.autocast(tb.type, dtype=torch.float16, enabled=tb.type == "cuda"):
                z = mh(b)[:, 0]
            l = mat_mat(z, b["nhan"])
            scaler.scale(l / k_tl).backward()
            if (i_b + 1) % k_tl == 0 or i_b + 1 == len(dl_train):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tong, n = tong + float(l.detach()) * len(b["nhan"]), n + len(b["nhan"])
            with torch.no_grad():
                y = b["nhan"]
                hl, du, that = y != NHAN_BO_QUA, z.detach() >= 0, y == 1
                dem += [int((du & that & hl).sum()), int((du & ~that & hl).sum()), int((~du & that & hl).sum())]
        csi = _danh_gia_val(mh, dl_val, tb)
        dc.append({"epoch": ep, "lr": opt.param_groups[0]["lr"], "loss_train": tong / n, "csi_train": float(dem[0] / max(dem.sum(), 1)), "csi_val": csi,
                   "giay": round(time.time() - t_ep, 1)})
        if lich:
            lich.step()
        print(f"{cfg['ten']} epoch {ep}: loss {tong / n:.4f} CSI val {csi:.4f} ({dc[-1]['giay']} s)", flush=True)
        if csi > tot:
            tot, khong_tien, epoch_tot = csi, 0, ep
            torch.save({"mo_hinh": m.state_dict(), "cfg": cfg, "epoch": ep, "csi_val": csi}, thu_muc / "best.pt")
        else:
            khong_tien += 1
            if khong_tien >= cfg.get("kien_nhan", 5):
                break
    ck = torch.load(thu_muc / "best.pt", map_location=tb, weights_only=False)
    m.load_state_dict(ck["mo_hinh"])
    _, z, y = _danh_gia_val(mh, dl_val, tb, lay_logit=True, rng=np.random.default_rng(cfg["seed"]))
    ck["nhiet_do"] = T = nhiet_do(z, y)
    torch.save(ck, thu_muc / "best.pt")
    ra = {"ten": cfg["ten"], "cfg": cfg, "duong_cong": dc, "epoch_tot": epoch_tot, "csi_val_tot": tot, "nhiet_do": T,
          "so_pixel_hieu_chinh": len(y), "thoi_gian_giay": round(time.time() - t_bd, 1), "thiet_bi": str(tb),
          "gpu": torch.cuda.get_device_name() if tb.type == "cuda" else None,
          "so_gpu": torch.cuda.device_count() if tb.type == "cuda" else 0,
          "gpu_dinh_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if tb.type == "cuda" else None,
          "git": chia_tap._git_hash()}
    DIR_KQ.mkdir(parents=True, exist_ok=True)
    f = DIR_KQ / f"{cfg['ten']}.json"
    f.write_text(json.dumps(ra, indent=1, ensure_ascii=False, default=str))
    return f
