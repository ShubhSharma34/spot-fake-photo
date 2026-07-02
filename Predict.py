"""
predict.py - Spot the Fake Photo
SalesCode AI Assignment

Usage:
    python predict.py some_image.jpg

Output:
    A single number from 0 to 1.
    0.0 = definitely a REAL photo
    1.0 = definitely a PHOTO OF A SCREEN (recapture)
"""

import sys
import os
import warnings
import numpy as np
import joblib
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CHANGE THIS to your actual folder path
BASE_DIR   = r"C:\Users\Shubh Sharma\Desktop\spot_fake"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
# ─────────────────────────────────────────────


def extract_features(image_path):
    """Extract classical CV features that distinguish real photos from screen recaptures."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img  = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}

    # 1. FFT — screen pixel grids create periodic frequency spikes
    f         = np.fft.fft2(gray)
    fshift    = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    features['fft_peak_ratio']     = float(magnitude.max() / (magnitude.mean() + 1e-6))
    features['fft_std']            = float(magnitude.std())
    h, w = magnitude.shape
    center_mask = np.zeros((h, w))
    center_mask[h//4:3*h//4, w//4:3*w//4] = 1
    features['fft_highfreq_ratio'] = float(
        magnitude[center_mask == 0].mean() / (magnitude[center_mask == 1].mean() + 1e-6)
    )

    # 2. Noise — real cameras have natural sensor noise; screens are too clean
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blur.astype(float)
    features['noise_std']      = float(noise.std())
    features['noise_kurtosis'] = float(stats.kurtosis(noise.flatten()))
    features['noise_mean_abs'] = float(np.abs(noise).mean())

    # 3. Local Binary Pattern — screen grids produce regular local textures
    lbp  = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for i, v in enumerate(hist):
        features[f'lbp_hist_{i}'] = float(v)
    features['lbp_uniformity'] = float((lbp == lbp.max()).mean())

    # 4. Color channel statistics
    for i, ch in enumerate(['b', 'g', 'r']):
        channel = img[:, :, i].astype(float)
        features[f'{ch}_mean'] = float(channel.mean())
        features[f'{ch}_std']  = float(channel.std())
        features[f'{ch}_skew'] = float(stats.skew(channel.flatten()))

    # 5. HSV saturation / value — screens have distinct backlight profiles
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['saturation_mean'] = float(hsv[:, :, 1].mean())
    features['saturation_std']  = float(hsv[:, :, 1].std())
    features['value_mean']      = float(hsv[:, :, 2].mean())
    features['value_std']       = float(hsv[:, :, 2].std())

    # 6. Edge sharpness — screens have softened pixel-rendered edges
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var']      = float(edges.var())
    features['laplacian_mean_abs'] = float(np.abs(edges).mean())
    features['laplacian_kurtosis'] = float(stats.kurtosis(edges.flatten()))

    # 7. Vignetting — real lenses darken corners; screens don't
    h, w = gray.shape
    center_val = float(gray[h//4:3*h//4, w//4:3*w//4].mean())
    corners    = [gray[:h//8, :w//8].mean(),  gray[:h//8, -w//8:].mean(),
                  gray[-h//8:, :w//8].mean(), gray[-h//8:, -w//8:].mean()]
    corner_val = float(np.mean(corners))
    features['vignette_ratio'] = float(center_val / (corner_val + 1e-6))
    features['vignette_diff']  = float(center_val - corner_val)

    # 8. Glare — screens produce specular bright spots from room lighting
    features['glare_ratio'] = float((gray > 240).mean())

    # 9. Color cast — screen backlights shift the R/B balance
    features['rb_ratio']   = float(img[:, :, 2].mean() / (img[:, :, 0].mean() + 1e-6))
    features['color_cast'] = float(img[:, :, 2].astype(float).mean() - img[:, :, 0].astype(float).mean())

    # 10. Block artifacts — JPEG re-compression on screen photos leaves 8×8 traces
    diff_h = np.abs(np.diff(gray.astype(float), axis=0))
    diff_v = np.abs(np.diff(gray.astype(float), axis=1))
    features['block_artifact_h'] = float(diff_h[7::8, :].mean() / (diff_h.mean() + 1e-6))
    features['block_artifact_v'] = float(diff_v[:, 7::8].mean() / (diff_v.mean() + 1e-6))

    return features


def predict(image_path, model_path=MODEL_PATH):
    """Return a score: 0 = real photo, 1 = photo of a screen."""
    model = joblib.load(model_path)
    feats = np.array(list(extract_features(image_path).values())).reshape(1, -1)
    score = model.predict_proba(feats)[0][1]
    return round(float(score), 4)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    score      = predict(image_path)
    print(score)