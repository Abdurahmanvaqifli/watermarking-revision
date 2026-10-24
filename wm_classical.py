#!/usr/bin/env python3
"""
Classical transform-domain watermarking experiments:
  (A) DCT-DWT       : 2-level Haar DWT -> LL2 band -> 4x4 block DCT -> QIM on DCT coefficient (0,1)
  (B) DCT-DWT-SVD   : 2-level Haar DWT -> LL2 band -> 4x4 block DCT -> SVD -> QIM on the largest singular value

Both schemes are BLIND: extraction needs only the watermarked/attacked image and the secret key
(key = block-permutation seed + quantisation step). Neither the original image nor the original watermark
is needed for extraction; the original watermark is used only to compute BER/NC.

Usage:
  python wm_classical.py --images /path/to/coco/val2017 --n 500 --out results --seed 2026

Outputs (in --out):
  config.json          all parameters, software versions, hardware
  image_ids.txt        identifiers of the images used, with their split
  per_image.csv        one row per (image, method, attack)
  summary.csv          mean / std / min / max per (method, attack)
  summary_table.md     compact table for the manuscript
"""
import argparse, json, os, platform, time, glob
import numpy as np
import cv2
import pandas as pd
from scipy.fft import dctn, idctn
from skimage.metrics import structural_similarity

# ----------------------------------------------------------------------------- transforms
def haar2(x):
    """One level of orthonormal 2-D Haar DWT. Returns LL, (LH, HL, HH)."""
    a = (x[0::2, :] + x[1::2, :]) / np.sqrt(2)
    d = (x[0::2, :] - x[1::2, :]) / np.sqrt(2)
    LL = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2)
    LH = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2)
    HL = (d[:, 0::2] + d[:, 1::2]) / np.sqrt(2)
    HH = (d[:, 0::2] - d[:, 1::2]) / np.sqrt(2)
    return LL, (LH, HL, HH)

def ihaar2(LL, bands):
    LH, HL, HH = bands
    h, w = LL.shape
    a = np.empty((h, 2 * w)); d = np.empty((h, 2 * w))
    a[:, 0::2] = (LL + LH) / np.sqrt(2); a[:, 1::2] = (LL - LH) / np.sqrt(2)
    d[:, 0::2] = (HL + HH) / np.sqrt(2); d[:, 1::2] = (HL - HH) / np.sqrt(2)
    x = np.empty((2 * h, 2 * w))
    x[0::2, :] = (a + d) / np.sqrt(2); x[1::2, :] = (a - d) / np.sqrt(2)
    return x

def dwt_ll(y, levels):
    details, LL = [], y
    for _ in range(levels):
        LL, b = haar2(LL); details.append(b)
    return LL, details

def idwt_ll(LL, details):
    for b in reversed(details):
        LL = ihaar2(LL, b)
    return LL

def blocks_of(LL, bs):
    h, w = LL.shape
    return [(r, c) for r in range(0, h, bs) for c in range(0, w, bs)]

# ----------------------------------------------------------------------------- method A: DCT-DWT
COEF = (0, 1)   # low-frequency AC coefficient of each 4x4 DCT block (horizontal frequency 1)

def qim_embed(x, bit, delta):
    """Quantisation index modulation: move x to the centre of the bit's sub-lattice cell."""
    return np.floor(x / delta) * delta + (0.75 if bit == 1 else 0.25) * delta

def qim_extract(x, delta):
    return 1 if (x / delta) % 1.0 >= 0.5 else 0

def embed_dct_dwt(y, bits, key, cfg):
    LL, det = dwt_ll(y, cfg["levels"])
    pos = blocks_of(LL, cfg["block"])
    order = np.random.default_rng(key).permutation(len(pos))[: len(bits)]
    for bit, k in zip(bits, order):
        r, c = pos[k]; bs = cfg["block"]
        D = dctn(LL[r:r+bs, c:c+bs], norm="ortho")
        D[COEF] = qim_embed(D[COEF], bit, cfg["delta_a"])
        LL[r:r+bs, c:c+bs] = idctn(D, norm="ortho")
    return idwt_ll(LL, det)

def extract_dct_dwt(y, nbits, key, cfg):
    LL, _ = dwt_ll(y, cfg["levels"])
    pos = blocks_of(LL, cfg["block"])
    order = np.random.default_rng(key).permutation(len(pos))[:nbits]
    out = np.empty(nbits, dtype=np.uint8)
    for i, k in enumerate(order):
        r, c = pos[k]; bs = cfg["block"]
        out[i] = qim_extract(dctn(LL[r:r+bs, c:c+bs], norm="ortho")[COEF], cfg["delta_a"])
    return out

# ----------------------------------------------------------------------------- method B: DCT-DWT-SVD
def embed_dct_dwt_svd(y, bits, key, cfg):
    LL, det = dwt_ll(y, cfg["levels"])
    pos = blocks_of(LL, cfg["block"])
    order = np.random.default_rng(key).permutation(len(pos))[: len(bits)]
    for bit, k in zip(bits, order):
        r, c = pos[k]; bs = cfg["block"]
        D = dctn(LL[r:r+bs, c:c+bs], norm="ortho")
        U, S, Vt = np.linalg.svd(D)
        S[0] = qim_embed(S[0], bit, cfg["delta_b"])                 # QIM on sigma_1
        LL[r:r+bs, c:c+bs] = idctn(U @ np.diag(S) @ Vt, norm="ortho")
    return idwt_ll(LL, det)

def extract_dct_dwt_svd(y, nbits, key, cfg):
    LL, _ = dwt_ll(y, cfg["levels"])
    pos = blocks_of(LL, cfg["block"])
    order = np.random.default_rng(key).permutation(len(pos))[:nbits]
    out = np.empty(nbits, dtype=np.uint8)
    for i, k in enumerate(order):
        r, c = pos[k]; bs = cfg["block"]
        S = np.linalg.svd(dctn(LL[r:r+bs, c:c+bs], norm="ortho"), compute_uv=False)
        out[i] = qim_extract(S[0], cfg["delta_b"])
    return out

METHODS = {
    "DCT-DWT":     (embed_dct_dwt, extract_dct_dwt),
    "DCT-DWT-SVD": (embed_dct_dwt_svd, extract_dct_dwt_svd),
}

# ----------------------------------------------------------------------------- colour handling
def embed_rgb(rgb, bits, key, cfg, method):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float64)
    ycc[..., 0] = METHODS[method][0](ycc[..., 0], bits, key, cfg)
    ycc = np.clip(np.round(ycc), 0, 255).astype(np.uint8)          # quantisation to 8 bits is part of the pipeline
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)

def extract_rgb(rgb, nbits, key, cfg, method):
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float64)
    return METHODS[method][1](y, nbits, key, cfg)

# ----------------------------------------------------------------------------- attacks
def jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def noise(img, sigma, rng):
    return np.clip(np.round(img + rng.normal(0, sigma, img.shape)), 0, 255).astype(np.uint8)

def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def crop_replace(img, frac):
    """Cropping attack: a corner region covering `frac` of the area is removed (set to 0); size is kept."""
    out = img.copy(); h, w = img.shape[:2]; s = np.sqrt(frac)
    out[: int(h * s), : int(w * s)] = 0
    return out

def make_attacks(rng):
    return {
        "None":                    lambda x: x,
        "JPEG Q=90":               lambda x: jpeg(x, 90),
        "JPEG Q=70":               lambda x: jpeg(x, 70),
        "JPEG Q=50":               lambda x: jpeg(x, 50),
        "Gaussian noise s=5":      lambda x: noise(x, 5, rng),
        "Gaussian noise s=10":     lambda x: noise(x, 10, rng),
        "Gaussian blur 5x5 s=1":   lambda x: cv2.GaussianBlur(x, (5, 5), 1.0),
        "Median 3x3":              lambda x: cv2.medianBlur(x, 3),
        "Scaling 0.5 -> 1":        lambda x: cv2.resize(cv2.resize(x, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
                                                        (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC),
        "Cropping 25%":            lambda x: crop_replace(x, 0.25),
        "Rotation 5 (realigned)":  lambda x: rotate(rotate(x, 5), -5),
        "Rotation 5 (no realign)": lambda x: rotate(x, 5),
        "Combined JPEG70+noise5":  lambda x: noise(jpeg(x, 70), 5, rng),
    }

# ----------------------------------------------------------------------------- metrics
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse), mse

def ber(w, w2):
    return float(np.mean(w != w2))

def nc(w, w2):
    """Normalised correlation on {0,1} watermarks: sum(w*w') / sqrt(sum(w^2) * sum(w'^2))."""
    w, w2 = w.astype(np.float64), w2.astype(np.float64)
    den = np.sqrt((w ** 2).sum() * (w2 ** 2).sum())
    return float((w * w2).sum() / den) if den > 0 else 0.0

def majority_baseline_ber(w):
    """BER of a trivial decoder that outputs the majority bit everywhere (i.e. ignores the image)."""
    p = w.mean(); return float(min(p, 1 - p))

# ----------------------------------------------------------------------------- data
def load_host(path, size):
    rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]; s = min(h, w)                               # centre crop to square, then resize
    rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

def split_ids(ids, seed, fracs=(0.70, 0.15, 0.15)):
    """Split by SOURCE IMAGE id, so that all versions of an image stay in one split (no leakage)."""
    ids = sorted(ids); perm = np.random.default_rng(seed).permutation(len(ids))
    n1 = int(fracs[0] * len(ids)); n2 = n1 + int(fracs[1] * len(ids))
    lab = {}
    for rank, i in enumerate(perm):
        lab[ids[i]] = "train" if rank < n1 else ("val" if rank < n2 else "test")
    return lab

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with host images (e.g. COCO val2017)")
    ap.add_argument("--n", type=int, default=500, help="number of images to sample")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--wm_side", type=int, default=32, help="watermark is wm_side x wm_side bits")
    ap.add_argument("--delta_a", type=float, default=48.0, help="DCT-DWT QIM step (embedding strength)")
    ap.add_argument("--delta_b", type=float, default=48.0, help="DCT-DWT-SVD QIM step (embedding strength)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--split", default="test", help="evaluate on this split (train/val/test/all)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.png")))
    rng = np.random.default_rng(a.seed)
    paths = [paths[i] for i in sorted(rng.choice(len(paths), size=min(a.n, len(paths)), replace=False))]
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    lab = split_ids(ids, a.seed)
    with open(os.path.join(a.out, "image_ids.txt"), "w") as f:
        for i in ids: f.write(f"{i}\t{lab[i]}\n")
    todo = [(p, i) for p, i in zip(paths, ids) if a.split == "all" or lab[i] == a.split]

    cfg = {"levels": 2, "block": 4, "delta_a": a.delta_a, "delta_b": a.delta_b}
    nbits = a.wm_side ** 2
    cap = (a.size // 2 ** cfg["levels"] // cfg["block"]) ** 2
    assert nbits <= cap, f"payload {nbits} > capacity {cap}"

    rows, attack_rng = [], np.random.default_rng(a.seed + 1)
    attacks = make_attacks(attack_rng)
    for n, (p, iid) in enumerate(todo):
        host = load_host(p, a.size)
        key = a.seed * 100003 + n                                   # per-image secret key
        w = np.random.default_rng(key + 7).integers(0, 2, nbits, dtype=np.uint8)   # unseen random watermark per image
        for m in METHODS:
            t0 = time.perf_counter(); wm = embed_rgb(host, w, key, cfg, m); t_emb = time.perf_counter() - t0
            p_db, mse = psnr(host, wm)
            ssim = structural_similarity(host, wm, channel_axis=2, data_range=255)
            for an, af in attacks.items():
                att = af(wm)
                t0 = time.perf_counter(); w2 = extract_rgb(att, nbits, key, cfg, m); t_ext = time.perf_counter() - t0
                rows.append(dict(image_id=iid, method=m, attack=an, psnr=p_db, ssim=ssim, mse=mse,
                                 ber=ber(w, w2), nc=nc(w, w2), baseline_ber=majority_baseline_ber(w),
                                 t_embed_ms=1000 * t_emb, t_extract_ms=1000 * t_ext))
        print(f"[{n+1}/{len(todo)}] {iid}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "per_image.csv"), index=False)
    agg = df.groupby(["method", "attack"], sort=False).agg(
        n=("ber", "size"), psnr_mean=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim_mean=("ssim", "mean"), ssim_std=("ssim", "std"),
        ber_mean=("ber", "mean"), ber_std=("ber", "std"), ber_min=("ber", "min"), ber_max=("ber", "max"),
        nc_mean=("nc", "mean"), nc_std=("nc", "std"), baseline_ber=("baseline_ber", "mean"),
        t_embed_ms=("t_embed_ms", "mean"), t_extract_ms=("t_extract_ms", "mean")).reset_index()
    agg.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    with open(os.path.join(a.out, "summary_table.md"), "w") as f:
        f.write("| Method | Attack | BER (mean ± std) | NC (mean ± std) |\n|---|---|---|---|\n")
        for _, r in agg.iterrows():
            f.write(f"| {r.method} | {r.attack} | {r.ber_mean:.4f} ± {r.ber_std:.4f} | {r.nc_mean:.4f} ± {r.nc_std:.4f} |\n")
        f.write("\n| Method | PSNR (dB) | SSIM | Embed (ms) | Extract (ms) |\n|---|---|---|---|---|\n")
        for m, g in agg[agg.attack == "None"].groupby("method", sort=False):
            r = g.iloc[0]
            f.write(f"| {m} | {r.psnr_mean:.2f} ± {r.psnr_std:.2f} | {r.ssim_mean:.4f} ± {r.ssim_std:.4f} "
                    f"| {r.t_embed_ms:.1f} | {r.t_extract_ms:.1f} |\n")

    json.dump({**vars(a), **cfg, "payload_bits": nbits, "capacity_bits": cap,
               "n_evaluated": len(todo), "dct_coefficient_method_a": COEF,
               "python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
               "cpu": platform.processor() or platform.machine(), "os": platform.platform()},
              open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(open(os.path.join(a.out, "summary_table.md")).read())

if __name__ == "__main__":
    main()
