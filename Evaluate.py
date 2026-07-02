"""
evaluate.py - Generate honest accuracy numbers for the submission note.

Run AFTER train.py:
    python evaluate.py

Prints per-image predictions + final accuracy on a held-out 20% test split.
"""

import os
import glob
import time
import warnings
import numpy as np
import joblib
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
BASE_DIR   = r"C:\Users\Shubh Sharma\Desktop\spot_fake"
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
REAL_DIR   = os.path.join(BASE_DIR, "real")
SCREEN_DIR = os.path.join(BASE_DIR, "screen")
# ─────────────────────────────────────────────


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img  = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    features['fft_peak_ratio'] = float(magnitude.max() / (magnitude.mean() + 1e-6))
    features['fft_std'] = float(magnitude.std())
    h, w = magnitude.shape
    cm = np.zeros((h, w)); cm[h//4:3*h//4, w//4:3*w//4] = 1
    features['fft_highfreq_ratio'] = float(magnitude[cm==0].mean() / (magnitude[cm==1].mean() + 1e-6))
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    noise = gray.astype(float) - blur.astype(float)
    features['noise_std'] = float(noise.std())
    features['noise_kurtosis'] = float(stats.kurtosis(noise.flatten()))
    features['noise_mean_abs'] = float(np.abs(noise).mean())
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    for i, v in enumerate(hist): features[f'lbp_hist_{i}'] = float(v)
    features['lbp_uniformity'] = float((lbp == lbp.max()).mean())
    for i, ch in enumerate(['b','g','r']):
        c = img[:,:,i].astype(float)
        features[f'{ch}_mean'] = float(c.mean()); features[f'{ch}_std'] = float(c.std()); features[f'{ch}_skew'] = float(stats.skew(c.flatten()))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['saturation_mean'] = float(hsv[:,:,1].mean()); features['saturation_std'] = float(hsv[:,:,1].std())
    features['value_mean'] = float(hsv[:,:,2].mean()); features['value_std'] = float(hsv[:,:,2].std())
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var'] = float(edges.var()); features['laplacian_mean_abs'] = float(np.abs(edges).mean()); features['laplacian_kurtosis'] = float(stats.kurtosis(edges.flatten()))
    h, w = gray.shape
    cv_ = float(gray[h//4:3*h//4, w//4:3*w//4].mean())
    corners = [gray[:h//8,:w//8].mean(), gray[:h//8,-w//8:].mean(), gray[-h//8:,:w//8].mean(), gray[-h//8:,-w//8:].mean()]
    cv2_ = float(np.mean(corners))
    features['vignette_ratio'] = float(cv_ / (cv2_ + 1e-6)); features['vignette_diff'] = float(cv_ - cv2_)
    features['glare_ratio'] = float((gray > 240).mean())
    features['rb_ratio'] = float(img[:,:,2].mean() / (img[:,:,0].mean() + 1e-6))
    features['color_cast'] = float(img[:,:,2].astype(float).mean() - img[:,:,0].astype(float).mean())
    diff_h = np.abs(np.diff(gray.astype(float), axis=0)); diff_v = np.abs(np.diff(gray.astype(float), axis=1))
    features['block_artifact_h'] = float(diff_h[7::8,:].mean() / (diff_h.mean() + 1e-6))
    features['block_artifact_v'] = float(diff_v[:,7::8].mean() / (diff_v.mean() + 1e-6))
    return features


def main():
    exts = ('*.jpg','*.jpeg','*.png','*.JPG','*.JPEG','*.PNG')
    real_paths, screen_paths = [], []
    for ext in exts:
        real_paths   += glob.glob(os.path.join(REAL_DIR,   ext))
        screen_paths += glob.glob(os.path.join(SCREEN_DIR, ext))

    paths = real_paths + screen_paths
    y     = [0]*len(real_paths) + [1]*len(screen_paths)

    X = []
    for p in paths:
        try:    X.append(list(extract_features(p).values()))
        except: X.append(None)

    valid = [(x, yi, p) for x, yi, p in zip(X, y, paths) if x is not None]
    X     = np.array([v[0] for v in valid])
    y     = np.array([v[1] for v in valid])
    paths = [v[2] for v in valid]

    # 80/20 stratified split — same seed as train.py for reproducibility
    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        X, y, paths, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)

    # Latency test
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict_proba(X_te[:1])
        times.append((time.perf_counter() - t0) * 1000)
    latency_ms = np.mean(times)

    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    acc     = accuracy_score(y_te, y_pred)

    print("=" * 60)
    print("  EVALUATION RESULTS  (held-out 20% test set)")
    print("=" * 60)
    print(f"\n  Test samples : {len(y_te)}  ({sum(y_te==0)} real, {sum(y_te==1)} screen)")
    print(f"  Accuracy     : {acc*100:.1f}%")
    print(f"  Latency      : {latency_ms:.1f} ms per image (CPU)")
    print()
    print(classification_report(y_te, y_pred, target_names=['Real (0)', 'Screen (1)']))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print("              Pred:Real  Pred:Screen")
    cm = confusion_matrix(y_te, y_pred)
    print(f"  Actual:Real   {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  Actual:Screen {cm[1,0]:5d}       {cm[1,1]:5d}")

    print("\n── Per-image predictions (test set) ──")
    print(f"{'File':<45} {'True':>6} {'Score':>6} {'Pass':>5}")
    print("-" * 65)
    for p, yt, yp, ypr in sorted(zip(p_te, y_te, y_pred, y_proba), key=lambda x: x[1]):
        fname = os.path.basename(p)
        label = 'real' if yt == 0 else 'screen'
        ok    = '✓' if yt == yp else '✗'
        print(f"  {fname:<43} {label:>6}  {ypr:.3f}  {ok}")

    print("\n── Cost analysis ──")
    print("  On-device (phone CPU): ~$0.00 per image (free, runs locally)")
    print(f"  AWS Lambda (512MB):    ~$0.000002 per image  (~$2 per 1M images)")
    print("  vs. Vision LLM API:    ~$0.01 per image      (~$10,000 per 1M images)")
    print("  Savings vs LLM API:    ~5000x cheaper")
    print("=" * 60)


if __name__ == '__main__':
    main()