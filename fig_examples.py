#!/usr/bin/env python3
"""
Example figure for the manuscript: host image, watermarked image (both methods), amplified difference,
and the extracted 32x32 watermark after JPEG Q=50.

Usage (Colab):
  python fig_examples.py --images val2017 --ids 000000000139 000000000285 000000000632 --out fig_examples.png
Choose test-split ids (see results/image_ids.txt) that do not show identifiable people.
"""
import argparse, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wm_classical import embed_rgb, extract_rgb, load_host, psnr, jpeg

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--ids", nargs="+", required=True)
ap.add_argument("--delta_a", type=float, default=48.0); ap.add_argument("--delta_b", type=float, default=48.0)
ap.add_argument("--out", default="fig_examples.png"); ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
cfg = {"levels": 2, "block": 4, "delta_a": a.delta_a, "delta_b": a.delta_b}

cols = ["Host image", "Watermarked\n(DCT-DWT)", "Watermarked\n(DCT-DWT-SVD)",
        "|Difference| x40\n(DCT-DWT-SVD)", "Extracted after JPEG Q=50\n(DCT-DWT-SVD)"]
fig, ax = plt.subplots(len(a.ids), 5, figsize=(15, 3.5 * len(a.ids)))
ax = np.atleast_2d(ax)
for r, iid in enumerate(a.ids):
    path = next(p for p in (os.path.join(a.images, iid + e) for e in (".jpg", ".png")) if os.path.exists(p))
    host = load_host(path, 512); key = a.seed * 100003 + r
    w = np.random.default_rng(key + 7).integers(0, 2, 1024, dtype=np.uint8)
    wa = embed_rgb(host, w, key, cfg, "DCT-DWT"); wb = embed_rgb(host, w, key, cfg, "DCT-DWT-SVD")
    diff = np.clip(np.abs(wb.astype(int) - host.astype(int)) * 40, 0, 255).astype(np.uint8)
    ext = extract_rgb(jpeg(wb, 50), 1024, key, cfg, "DCT-DWT-SVD")
    ber = np.mean(ext != w)
    # 3-colour map of extracted bits: correct 0 = black, correct 1 = white, error = red
    vis = np.zeros((1024, 3), np.uint8); vis[ext == 1] = 255; vis[ext != w] = [220, 30, 30]
    panels = [host, wa, wb, diff, vis.reshape(32, 32, 3)]
    titles = [f"COCO id {int(iid) if iid.isdigit() else iid}", f"PSNR {psnr(host, wa)[0]:.2f} dB", f"PSNR {psnr(host, wb)[0]:.2f} dB",
              "", f"BER {ber:.4f} (errors in red)"]
    for c in range(5):
        ax[r, c].imshow(panels[c], interpolation="nearest" if c == 4 else None)
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        ax[r, c].set_xlabel(titles[c], fontsize=10)
        if r == 0: ax[r, c].set_title(cols[c], fontsize=11)
plt.tight_layout(h_pad=2.0); plt.savefig(a.out, dpi=300); print("saved", a.out)
