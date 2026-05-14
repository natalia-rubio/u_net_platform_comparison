#!/usr/bin/env python3
"""
Generate 256x256 4-bit (16-level) grayscale images with a hollow ellipse,
plus a binary black-and-white label mask per sample.

Requires Python 3.6+ (use ``python3`` if ``python`` is 2.x on your system).

Background is always darker (lower gray level) than the oval.
Writes matching filenames: <out>/images/oval_XXXX.png (grayscale),
<out>/labels/oval_XXXX.png (BW: 0 = bg, 255 = oval). Default <out> is data/.

Random parameters per image:
  - Semi-major and semi-minor axes (ellipse fits comfortably in the frame)
  - Rotation in degrees [0, 45]
  - Annulus thickness (edge width) in pixels [2, 6]

Grayscale images are further degraded with random Gaussian blur and Gaussian noise;
label masks stay sharp 0/255.
"""

import sys

if sys.version_info < (3, 6):
    sys.stderr.write(
        "generate_data.py needs Python 3.6+ (you have %s). "
        "Try: python3 generate_data.py\n" % sys.version.split()[0]
    )
    sys.exit(1)

import argparse
import math
import random
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
IMAGE_SIZE = 256
CENTER = (IMAGE_SIZE - 1) / 2.0
LEVELS = 16  # 4-bit grayscale


def random_oval_params(rng):
    """Semi-major a, semi-minor b, rotation radians, edge width (pixels)."""
    # Ranges chosen so ovals stay inside the canvas with margin.
    a = rng.uniform(55.0, 105.0)
    b = rng.uniform(35.0, min(a - 5.0, 95.0))
    if b > a:
        a, b = b, a
    edge = rng.randint(2, 15)
    # Shrink outer ellipse if inner axes would be invalid.
    a = max(a, float(edge) + 8.0)
    b = max(b, float(edge) + 8.0)
    deg = rng.uniform(0.0, 180.0)
    theta = math.radians(deg)
    return a, b, theta, edge


def hollow_oval_mask(a, b, theta, edge):
    """
    Boolean mask (H, W): True on the hollow oval (annulus between two ellipses).

    Ellipse centered on image center; major axis along x before rotation by theta.
    """
    y = np.arange(IMAGE_SIZE, dtype=np.float64)[:, None]
    x = np.arange(IMAGE_SIZE, dtype=np.float64)[None, :]
    cx, cy = CENTER, CENTER
    dx = x - cx
    dy = y - cy

    c = math.cos(theta)
    s = math.sin(theta)
    # Rotate coordinates by -theta: express point in ellipse-aligned frame.
    xr = dx * c + dy * s
    yr = -dx * s + dy * c

    ai = max(a - edge, 1e-6)
    bi = max(b - edge, 1e-6)

    f_outer = (xr / a) ** 2 + (yr / b) ** 2
    f_inner = (xr / ai) ** 2 + (yr / bi) ** 2

    return (f_outer <= 1.0) & (f_inner >= 1.0)


def to_4bit_gray(arr_0_15):
    """Map integer labels 0..15 to uint8 0..255 (standard 4-bit LSB scaling)."""
    return (np.clip(arr_0_15, 0, LEVELS - 1).astype(np.uint8) * 17).clip(0, 255)


def add_blur_and_noise(gray, rng):
    """
    Mild Gaussian blur plus additive Gaussian noise in [0, 255] space.
    Labels are not modified; only saved images are degraded.
    """
    radius = rng.uniform(0.4, 2.5)
    pil = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=radius))
    x = np.asarray(pil, dtype=np.float32)
    sigma = rng.uniform(1.5, 8.0)
    noise_prng = np.random.default_rng(rng.randrange(2**32))
    x += noise_prng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def render_pair(rng):
    """
    Return (grayscale_uint8, label_uint8).

    Grayscale uses 16 levels only; background level is strictly less than oval.
    Label is binary: 0 background, 255 on the hollow oval.
    """
    a, b, theta, edge = random_oval_params(rng)

    bg = rng.randint(0, LEVELS - 2)
    fg = rng.randint(bg + 1, LEVELS - 1)

    levels = np.full((IMAGE_SIZE, IMAGE_SIZE), bg, dtype=np.int32)
    mask = hollow_oval_mask(a, b, theta, edge)
    levels[mask] = fg

    gray = to_4bit_gray(levels)
    label = np.where(mask, np.uint8(255), np.uint8(0))
    return gray, label

def transform_data():
    image_files = list(Path("data/images").glob("*.png"))
    label_files = list(Path("data/labels").glob("*.png"))
    assert len(image_files) == len(label_files)
    images = [transforms.ToTensor()(Image.open(file)).reshape(1, 1, 256, 256) for file in image_files]
    labels = [transforms.ToTensor()(Image.open(file)).reshape(1, 1, 256, 256) for file in label_files]
    torch.save(torch.cat(images, dim=0), "data/images.pt")
    torch.save(torch.cat(labels, dim=0), "data/labels.pt")
    print("Transformed {} images and {} labels".format(len(images), len(labels)))
    print(images[0].shape, labels[0].shape)
    return

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-n", "--count", type=int, default=100, help="Number of images")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("data"),
        help="Output root; grayscale -> <out>/images/, labels -> <out>/labels/",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed (optional)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    images_dir = args.out / "images"
    labels_dir = args.out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        gray, label = render_pair(rng)
        gray = add_blur_and_noise(gray, rng)
        name = "oval_{:04d}.png".format(i)
        Image.fromarray(gray).save(images_dir / name)
        Image.fromarray(label).save(labels_dir / name)

    print(
        "Wrote {} pair(s) to {} and {} ({} files)".format(
            args.count,
            images_dir.resolve(),
            labels_dir.resolve(),
            args.count * 2,
        )
    )

    transform_data()


if __name__ == "__main__":
    main()
