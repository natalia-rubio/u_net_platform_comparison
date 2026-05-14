# U-Net platform comparison (synthetic ovals)

Small PyTorch project that generates **256×256** synthetic grayscale images with **hollow ellipses**, trains a compact **U-Net** to predict segmentation masks, and includes a **benchmark script** that sweeps hyperparameters and plots timings with and without GPU acceleration.

## Requirements

- **Python 3.6+** (use `python3` on systems where `python` is still 2.x).
- **PyTorch** and **torchvision** (same versions you use for other projects are fine).
- **NumPy**, **Pillow**, **Matplotlib** (for `test_performance.py`).

Install in your environment, for example:

```bash
pip install torch torchvision numpy pillow matplotlib
```

## Repository layout

| Path | Purpose |
|------|---------|
| `generate_data.py` | Create PNG pairs + rebuild `images.pt` / `labels.pt` |
| `data/images/` | Grayscale inputs (degraded with blur + noise) |
| `data/labels/` | Binary masks (0 / 255 in PNG) |
| `data/images.pt`, `data/labels.pt` | Stacked tensors used by training (`torch.load`) |
| `torch_unet.py` | U-Net definition, `load_data()`, `train_model()` |
| `test_performance.py` | Runs experiments, saves plots and `results/results.json` |

## 1. Generate data

Creates `oval_0000.png` … under `data/images` and `data/labels`, then builds `data/images.pt` and `data/labels.pt` from **all** matching PNGs (sorted by filename).

```bash
python3 generate_data.py -n 100 --seed 0
```

Useful flags:

- **`-n` / `--count`** — number of synthetic pairs to write (default 100).
- **`-o` / `--out`** — root directory (default `data/`; PNGs go to `<out>/images` and `<out>/labels`, tensors to `<out>/images.pt` and `<out>/labels.pt`).
- **`--transform-only`** — only rebuild the `.pt` files from existing PNGs (no new images). Use this after you add or change PNGs manually.

Training reads **`data/images.pt`** and **`data/labels.pt`**, not the PNG folders directly. Regenerate or use `--transform-only` whenever the PNG set changes.

**Labels:** PNGs are 0 or 255. After `ToTensor()` in the saved tensors, values are **0.0** and **1.0** (float).

## 2. Train the U-Net

From the repo root (so `data/*.pt` paths resolve):

```bash
python3 torch_unet.py
```

`train_model()` in `torch_unet.py`:

- Loads the full tensors, then **shuffles** and splits into train and test set internally.
- For each run, it **subsamples** that split: training uses the first `num_batches * batch_size` samples of the train split; test uses the first `batch_size` samples of the test split. That keeps benchmark runs small and comparable.
- **`use_gpu=True`** moves the model and tensors to CUDA. You need a **PyTorch build with CUDA**; CPU-only wheels will raise an error on `.to("cuda")`.

## 3. Benchmarks

```bash
python3 test_performance.py
```

Runs three sweeps (kernel size, batch size, number of batches), compares CPU vs GPU timing where applicable, writes PDF/PNG under `results/`, and saves `results/results.json` (each experiment includes an `'x'` list for the horizontal axis).

---

If something does not match your cluster (paths, CUDA, Python command), adjust the commands or paths above; the code assumes a current working directory of the project root.
