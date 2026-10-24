#!/usr/bin/env python3
"""
Neural-network-based watermark EXTRACTION in the transform domain.

Task definition
  Input : a (possibly attacked) watermarked RGB image + the secret key (block permutation seed, QIM step).
  Output: the 1024 embedded watermark bits (one probability per bit, thresholded at 0.5).
  The network does NOT embed and does NOT need the original image or watermark.

Pipeline per bit (block selected by the key):
  1. Same transforms as embedding: Y channel -> 2-level Haar DWT -> LL2 -> 4x4 block DCT (-> SVD for method B).
  2. Embedding value v (DCT coefficient (0,1) for DCT-DWT, sigma_1 for DCT-DWT-SVD), normalised x = v / delta.
  3. Image-level lattice re-synchronisation: estimate the gain g and offset s that best restore the QIM lattice
     (maximum of |mean exp(i*4*pi*g*x)| over all blocks).  This compensates amplitude changes (blur, JPEG)
     and DC shifts (noise, brightness).
  4. Features: QIM phase before/after re-synchronisation, log-magnitudes of the 16 DCT coefficients of the block
     (local texture / reliability), and image-level context (g, s, lattice coherence, mean block energy).
  5. A multilayer perceptron (shared across all blocks) outputs P(bit = 1).

Three decoders are evaluated on identical test data (ablation):
  classical      : plain QIM decision on x                      (= wm_classical.py)
  classical+sync : plain QIM decision on the re-synchronised value
  neural network : MLP on the features above

Training uses ONLY the train split (random attacks with randomised parameters); model selection (epoch) uses
ONLY the val split; reported numbers use ONLY the test split with the fixed attack list of wm_classical.py.
Splitting is by source image id, so no image appears in more than one split.

Usage:
  python wm_learned.py --images /path/to/coco/val2017 --n 1000 --out results_nn --seed 2026
"""
import argparse, copy, json, os, platform, time, glob
import numpy as np, cv2, pandas as pd, joblib, sklearn
from scipy.fft import dctn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from wm_classical import (dwt_ll, blocks_of, embed_rgb, load_host, split_ids, make_attacks,
                          jpeg, noise, rotate, crop_replace, COEF)

GAIN_GRID = {"DCT-DWT": np.linspace(0.5, 1.1, 61), "DCT-DWT-SVD": np.linspace(0.97, 1.03, 61)}

# ----------------------------------------------------------------------------- feature extraction
def block_values(rgb, nbits, key, cfg, method):
    """Return x = v/delta (nbits,) and log-magnitude DCT spectra (nbits, 16) of the keyed blocks."""
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float64)
    LL, _ = dwt_ll(y, cfg["levels"]); bs = cfg["block"]
    pos = blocks_of(LL, bs)
    order = np.random.default_rng(key).permutation(len(pos))[:nbits]
    delta = cfg["delta_a"] if method == "DCT-DWT" else cfg["delta_b"]
    x = np.empty(nbits); spec = np.empty((nbits, bs * bs))
    for i, k in enumerate(order):
        r, c = pos[k]
        D = dctn(LL[r:r+bs, c:c+bs], norm="ortho")
        v = D[COEF] if method == "DCT-DWT" else np.linalg.svd(D, compute_uv=False)[0]
        x[i] = v / delta; spec[i] = np.log1p(np.abs(D).ravel())
    return x, spec

def resync(x, method, valid):
    """Estimate gain g and offset s restoring the QIM lattice; return corrected y and context.
    Only blocks with non-zero content (`valid`) are used, so cropped/blanked regions do not bias the estimate."""
    G = GAIN_GRID[method]
    xv = x[valid] if valid.sum() >= 16 else x
    z = np.exp(1j * 4 * np.pi * np.outer(G, xv))          # cells at 0.25/0.75 map to angle pi
    m = z.mean(axis=1); coh = np.abs(m)
    j = int(np.argmax(coh)); g = G[j]
    s = np.angle(-m[j]) / (4 * np.pi)                     # lattice offset in cell units, wrapped to (-1/4, 1/4]
    y = g * x - s
    c1 = float(np.abs(np.exp(1j * 4 * np.pi * xv).mean()))
    return y, np.array([g, s, coh[j], c1, valid.mean()])

def features(rgb, nbits, key, cfg, method):
    x, spec = block_values(rgb, nbits, key, cfg, method)
    valid = np.expm1(spec).sum(axis=1) > 1.0          # block is not blank (e.g. removed by cropping)
    y, ctx = resync(x, method, valid)
    px, py = 2 * np.pi * x, 2 * np.pi * y
    ctx_all = np.concatenate([ctx, [spec.mean()]])
    F = np.column_stack([np.sin(px), np.cos(px), np.sin(py), np.cos(py), spec,
                         np.repeat(ctx_all[None, :], nbits, axis=0)])
    return F, x, y

def qim_bits(v):
    return ((v % 1.0) >= 0.5).astype(np.uint8)

# ----------------------------------------------------------------------------- randomised training attacks
def random_attack(img, rng):
    k = rng.integers(0, 9)
    if k == 0: return img, "none"
    if k == 1: return jpeg(img, int(rng.integers(40, 96))), "jpeg"
    if k == 2: return noise(img, rng.uniform(1, 12), rng), "noise"
    if k == 3:
        s = rng.uniform(0.3, 1.5); return cv2.GaussianBlur(img, (5, 5), s), "blur"
    if k == 4: return cv2.medianBlur(img, 3), "median"
    if k == 5:
        f = rng.uniform(0.4, 0.9); h, w = img.shape[:2]
        sm = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC), "scale"
    if k == 6: return crop_replace(img, rng.uniform(0.05, 0.35)), "crop"
    if k == 7:
        a = rng.uniform(-8, 8); return rotate(rotate(img, a), -a), "rot_realigned"
    return noise(jpeg(img, int(rng.integers(50, 90))), rng.uniform(1, 8), rng), "combined"

# ----------------------------------------------------------------------------- data building
def build_set(items, cfg, method, nbits, draws, per_img, rng, seed):
    X, Y = [], []
    for n, (p, iid, key) in enumerate(items):
        host = load_host(p, cfg["size"])
        for d in range(draws):
            w = rng.integers(0, 2, nbits, dtype=np.uint8)
            wm = embed_rgb(host, w, key + 1000 * (d + 1), cfg, method)
            att, _ = random_attack(wm, rng)
            F, _, _ = features(att, nbits, key + 1000 * (d + 1), cfg, method)
            idx = rng.choice(nbits, size=min(per_img, nbits), replace=False)
            X.append(F[idx]); Y.append(w[idx])
    return np.vstack(X).astype(np.float32), np.concatenate(Y)

def train_mlp(Xtr, Ytr, Xva, Yva, a, log):
    sc = StandardScaler().fit(Xtr); Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    mlp = MLPClassifier(hidden_layer_sizes=tuple(a.hidden), activation="relu", solver="adam",
                        learning_rate_init=a.lr, batch_size=a.batch, alpha=a.l2, random_state=a.train_seed)
    rng = np.random.default_rng(a.train_seed); best, best_ber = None, 1.0
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(Xtr))
        for b in range(0, len(perm), 65536):                 # partial_fit over shuffled chunks = one epoch
            sl = perm[b:b + 65536]; mlp.partial_fit(Xtr[sl], Ytr[sl], classes=[0, 1])
        vb = float(np.mean(mlp.predict(Xva) != Yva)); log.append(dict(epoch=ep, val_ber=vb, train_loss=mlp.loss_))
        print(f"  epoch {ep:3d}  train loss {mlp.loss_:.4f}  val BER {vb:.4f}", flush=True)
        if vb < best_ber: best, best_ber = copy.deepcopy(mlp), vb
    return sc, best, best_ber

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=512); ap.add_argument("--wm_side", type=int, default=32)
    ap.add_argument("--delta_a", type=float, default=48.0); ap.add_argument("--delta_b", type=float, default=48.0)
    ap.add_argument("--draws", type=int, default=6, help="random attacks per training image")
    ap.add_argument("--per_img", type=int, default=256, help="blocks sampled per training draw")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--l2", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026, help="data seed: image subset, split, test keys/watermarks, test attacks")
    ap.add_argument("--train_seed", type=int, default=None,
                    help="training seed: training watermarks/keys/attacks, weight initialisation, shuffling "
                         "(default = --seed, which reproduces the original run)")
    ap.add_argument("--out", default="results_nn")
    ap.add_argument("--methods", nargs="+", default=["DCT-DWT", "DCT-DWT-SVD"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.train_seed is None: a.train_seed = a.seed
    key_off = 0 if a.train_seed == a.seed else a.train_seed * 1000003   # different training keys for repeated runs

    paths = sorted(glob.glob(os.path.join(a.images, "*.jpg")) + glob.glob(os.path.join(a.images, "*.png")))
    rng = np.random.default_rng(a.seed)
    paths = [paths[i] for i in sorted(rng.choice(len(paths), size=min(a.n, len(paths)), replace=False))]
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    lab = split_ids(ids, a.seed)                                      # identical split to wm_classical.py
    items = {s: [(p, i, a.seed * 100003 + n) for n, (p, i) in enumerate(zip(paths, ids)) if lab[i] == s]
             for s in ("train", "val", "test")}
    with open(os.path.join(a.out, "image_ids.txt"), "w") as f:
        for i in ids: f.write(f"{i}\t{lab[i]}\n")
    cfg = {"levels": 2, "block": 4, "delta_a": a.delta_a, "delta_b": a.delta_b, "size": a.size}
    nbits = a.wm_side ** 2
    rows, logs, info = [], [], {}

    for m in a.methods:
        print(f"== {m}: building training/validation sets", flush=True)
        t0 = time.perf_counter()
        train_items = [(p, i, k + key_off) for (p, i, k) in items["train"]]
        Xtr, Ytr = build_set(train_items, cfg, m, nbits, a.draws, a.per_img, np.random.default_rng(a.train_seed + 11), a.seed)
        Xva, Yva = build_set(items["val"], cfg, m, nbits, 2, nbits, np.random.default_rng(a.seed + 12), a.seed)
        print(f"   train {Xtr.shape}, val {Xva.shape}", flush=True)
        log = []; sc, mlp, vb = train_mlp(Xtr, Ytr, Xva, Yva, a, log)
        for r in log: logs.append(dict(method=m, **r))
        info[m] = dict(train_rows=int(len(Xtr)), val_rows=int(len(Xva)), n_features=int(Xtr.shape[1]),
                       best_val_ber=vb, best_epoch=int(np.argmin([r["val_ber"] for r in log]) + 1),
                       n_parameters=int(sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
                       train_time_s=time.perf_counter() - t0)
        joblib.dump({"scaler": sc, "mlp": mlp, "cfg": cfg, "method": m}, os.path.join(a.out, f"model_{m}.joblib"))

        print(f"== {m}: testing", flush=True)
        attacks = make_attacks(np.random.default_rng(a.seed + 1))
        for (p, iid, key) in items["test"]:
            host = load_host(p, a.size)
            w = np.random.default_rng(key + 7).integers(0, 2, nbits, dtype=np.uint8)
            wm = embed_rgb(host, w, key, cfg, m)
            for an, af in attacks.items():
                att = af(wm)
                t0 = time.perf_counter(); F, x, yv = features(att, nbits, key, cfg, m)
                pred = mlp.predict(sc.transform(F.astype(np.float32))); t_nn = time.perf_counter() - t0
                for dec, bits in (("classical", qim_bits(x)), ("classical+sync", qim_bits(yv)), ("neural network", pred)):
                    rows.append(dict(image_id=iid, method=m, attack=an, decoder=dec,
                                     ber=float(np.mean(bits != w)), t_extract_ms=1000 * t_nn if dec == "neural network" else np.nan))

    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "per_image.csv"), index=False)
    pd.DataFrame(logs).to_csv(os.path.join(a.out, "training_log.csv"), index=False)
    piv = df.groupby(["method", "attack", "decoder"], sort=False)["ber"].agg(["mean", "std"]).reset_index()
    piv.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    with open(os.path.join(a.out, "summary_table.md"), "w") as f:
        decs = ["classical", "classical+sync", "neural network"]
        f.write("| Method | Attack | " + " | ".join(f"BER {d}" for d in decs) + " |\n|---|---|" + "---|" * 3 + "\n")
        for (m, an), g in piv.groupby(["method", "attack"], sort=False):
            vals = {r.decoder: f"{r['mean']:.4f} ± {r['std']:.4f}" for _, r in g.iterrows()}
            f.write(f"| {m} | {an} | " + " | ".join(vals[d] for d in decs) + " |\n")
    json.dump({**vars(a), **cfg, "payload_bits": nbits, "gain_grid": {k: [float(v.min()), float(v.max()), len(v)] for k, v in GAIN_GRID.items()},
               "n_train_images": len(items["train"]), "n_val_images": len(items["val"]), "n_test_images": len(items["test"]),
               "models": info, "python": platform.python_version(), "sklearn": sklearn.__version__,
               "numpy": np.__version__, "opencv": cv2.__version__, "cpu": platform.processor() or platform.machine(),
               "os": platform.platform()}, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    print(open(os.path.join(a.out, "summary_table.md")).read())

if __name__ == "__main__":
    main()
