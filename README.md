# Robust Digital Image Watermarking Using Transform-Domain Embedding and Neural Network-Based Extraction

Code, configuration, trained decoders and per-image measurements for the paper:

> A. Vagifli, "Robust Digital Image Watermarking Using Transform-Domain Embedding and Neural Network-Based Extraction" (under review).

## Contents

| File | Description |
|---|---|
| `wm_classical.py` | DCT–DWT and DCT–DWT–SVD embedding (QIM), attacks, classical blind extraction, PSNR/SSIM/BER/NC, timing |
| `wm_learned.py` | Lattice resynchronisation and neural network (MLP) decoder: training, validation-based epoch selection, test evaluation |
| `fig_examples.py` | Generates the example figure (Fig. 2) |
| `wm_colab.ipynb` | Notebook that runs everything on Google Colab (CPU only) |
| `image_ids.txt` | The 1,000 MS COCO val2017 image identifiers used and their train/val/test split (identical for both runs) |
| `config.json` | All parameters, software versions and hardware of the classical run (`wm_classical.py`) |
| `per_image.csv` | Classical run: per-image, per-method, per-attack PSNR, SSIM, BER, NC and timings (Tables 2–3) |
| `summary.csv`, `summary_table.md` | Classical run: mean, standard deviation, minimum and maximum per method and attack |
| `config_nn.json` | Neural network run (`wm_learned.py`): all parameters, training set sizes, selected epochs |
| `per_image_nn.csv` | Neural network run: per-image BER of the three decoders (Table 4, Fig. 3) |
| `summary_nn.csv`, `summary_table_nn.md` | Neural network run: mean and standard deviation per method, attack and decoder |
| `training_log.csv` | Training loss and validation BER per epoch (Fig. 4) |
| `model_DCT-DWT.joblib`, `model_DCT-DWT-SVD.joblib` | Trained decoders (scikit-learn scaler + MLP); load with `joblib.load` |
| `stats_method_comparison.csv` | Paired Wilcoxon tests, method A vs. method B (Holm-corrected) |
| `stats_nn_vs_classical.csv` | Paired Wilcoxon tests, neural network vs. classical decoder (Holm-corrected) |
| `results_nn_s2027/`, `results_nn_s2028/` | Repeated training runs of the neural network decoder with training seeds 2027 and 2028 (`per_image.csv`, `config.json`, `training_log.csv`) |
| `stats_nn_vs_classical_3runs.csv` | Decoder comparison over the three training runs (Table 4): run-averaged BER, SD across runs, Holm-corrected tests, number of significant runs |
| `training_log_3runs.csv` | Training loss and validation BER of all three runs (Fig. 4) |

The scripts write their outputs to the folder given by `--out` (`results` and `results_nn` in the commands below); the files above are those outputs, renamed with the suffix `_nn` for the neural network run.

## Data

Host images are from the MS COCO 2017 validation set (https://cocodataset.org/, `val2017.zip`). The images are not redistributed here; `image_ids.txt` lists the identifiers used. The subset and the split are reproduced deterministically from the random seed.

## Reproducing the results

```bash
pip install numpy scipy opencv-python-headless scikit-image pandas scikit-learn joblib
wget http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip

python wm_classical.py --images val2017 --n 1000 --out results       # Tables 2-3 (minutes)
python wm_learned.py   --images val2017 --n 1000 --out results_nn    # Table 4, Figs. 3-4 (about 1 h on CPU)
python wm_learned.py   --images val2017 --n 1000 --train_seed 2027 --out results_nn_s2027   # repeated training runs
python wm_learned.py   --images val2017 --n 1000 --train_seed 2028 --out results_nn_s2028
```

Default parameters: 512×512 images, 2-level Haar DWT, 4×4 blocks of the LL2 band, 1024-bit random watermark per image, quantisation step Δ = 48 for both methods, seed 2026, split 700/150/150 by image identifier. No GPU is required.

`--seed` fixes the image subset, the split and all test conditions; `--train_seed` changes only the training of the neural network decoder (default: equal to `--seed`).

Timings depend on the hardware. The first run used OpenCV 4.14.0 and the repeated runs OpenCV 5.0.0; the two versions give identical results except for the rotation attacks, where the interpolation differs slightly (mean BER differences ≤ 0.003). To reproduce the first run exactly, install OpenCV 4.14 (`pip install "opencv-python-headless~=4.14.0"`).

## License

MIT License (see `LICENSE`).
