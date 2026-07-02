"""
train.py - Spot the Fake Photo
SalesCode AI Assignment

Run: python train.py
Output: model.pkl (saved in same folder)
"""

import os
import glob
import warnings
import numpy as np
import joblib
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CHANGE THIS to your actual folder path
BASE_DIR = r"C:\Users\Shubh Sharma\Desktop\spot_fake"
# ─────────────────────────────────────────────

REAL_DIR   = os.path.join(BASE_DIR, "real")
SCREEN_DIR = os.path.join(BASE_DIR, "screen")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def extract_features(image_path):
    """Extract classical CV features that distinguish real photos from screen recaptures."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img  = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = {}

    # ── 1. FFT Analysis ──────────────────────────────────────────────────────
    # Screen pixel grids create regular frequency spikes in Fourier domain
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

    # ── 2. Noise Analysis ────────────────────────────────────────────────────
    # Real camera sensors introduce natural Gaussian noise.
    # Screens are "too clean" — very low noise_std.
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray.astype(float) - blur.astype(float)
    features['noise_std']      = float(noise.std())
    features['noise_kurtosis'] = float(stats.kurtosis(noise.flatten()))
    features['noise_mean_abs'] = float(np.abs(noise).mean())

    # ── 3. Local Binary Pattern ──────────────────────────────────────────────
    # Screen pixel grids create highly regular local textures
    lbp  = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for i, v in enumerate(hist):
        features[f'lbp_hist_{i}'] = float(v)
    features['lbp_uniformity'] = float((lbp == lbp.max()).mean())

    # ── 4. Color Channel Statistics ──────────────────────────────────────────
    for i, ch in enumerate(['b', 'g', 'r']):
        channel = img[:, :, i].astype(float)
        features[f'{ch}_mean'] = float(channel.mean())
        features[f'{ch}_std']  = float(channel.std())
        features[f'{ch}_skew'] = float(stats.skew(channel.flatten()))

    # ── 5. HSV Saturation & Value ────────────────────────────────────────────
    # Screens often have different saturation profiles due to backlight
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['saturation_mean'] = float(hsv[:, :, 1].mean())
    features['saturation_std']  = float(hsv[:, :, 1].std())
    features['value_mean']      = float(hsv[:, :, 2].mean())
    features['value_std']       = float(hsv[:, :, 2].std())

    # ── 6. Edge Sharpness ────────────────────────────────────────────────────
    # Screens have softened/filtered edges due to sub-pixel rendering
    # Real photos have natural edge distributions
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var']      = float(edges.var())
    features['laplacian_mean_abs'] = float(np.abs(edges).mean())
    features['laplacian_kurtosis'] = float(stats.kurtosis(edges.flatten()))

    # ── 7. Vignetting ────────────────────────────────────────────────────────
    # Real camera lenses cause natural darkening at corners (vignetting).
    # Screens + camera combo produces different corner-to-center ratios.
    h, w = gray.shape
    center_val = float(gray[h//4:3*h//4, w//4:3*w//4].mean())
    corners    = [gray[:h//8, :w//8].mean(),  gray[:h//8, -w//8:].mean(),
                  gray[-h//8:, :w//8].mean(), gray[-h//8:, -w//8:].mean()]
    corner_val = float(np.mean(corners))
    features['vignette_ratio'] = float(center_val / (corner_val + 1e-6))
    features['vignette_diff']  = float(center_val - corner_val)

    # ── 8. Glare Detection ───────────────────────────────────────────────────
    # Screens often produce specular glare (very bright spots)
    features['glare_ratio'] = float((gray > 240).mean())

    # ── 9. Color Cast ────────────────────────────────────────────────────────
    # Screen backlights often produce a blue/cold tint
    features['rb_ratio']    = float(img[:, :, 2].mean() / (img[:, :, 0].mean() + 1e-6))
    features['color_cast']  = float(img[:, :, 2].astype(float).mean() - img[:, :, 0].astype(float).mean())

    # ── 10. Block Artifact Detection ─────────────────────────────────────────
    # JPEG re-compression of screen photos creates 8x8 block artifacts
    diff_h = np.abs(np.diff(gray.astype(float), axis=0))
    diff_v = np.abs(np.diff(gray.astype(float), axis=1))
    features['block_artifact_h'] = float(diff_h[7::8, :].mean() / (diff_h.mean() + 1e-6))
    features['block_artifact_v'] = float(diff_v[:, 7::8].mean() / (diff_v.mean() + 1e-6))

    return features


def load_dataset(real_dir, screen_dir):
    """Load all images and extract features."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    real_paths   = []
    screen_paths = []

    for ext in exts:
        real_paths   += glob.glob(os.path.join(real_dir,   ext))
        screen_paths += glob.glob(os.path.join(screen_dir, ext))

    print(f"Found {len(real_paths)} real photos and {len(screen_paths)} screen photos")

    X, y, failed = [], [], []

    for path in real_paths:
        try:
            feats = extract_features(path)
            X.append(list(feats.values()))
            y.append(0)
        except Exception as e:
            failed.append((path, str(e)))

    for path in screen_paths:
        try:
            feats = extract_features(path)
            X.append(list(feats.values()))
            y.append(1)
        except Exception as e:
            failed.append((path, str(e)))

    if failed:
        print(f"\nWarning: {len(failed)} images could not be loaded:")
        for p, e in failed:
            print(f"  {p}: {e}")

    return np.array(X), np.array(y)


def train(real_dir, screen_dir, model_path):
    print("=" * 55)
    print("  Spot the Fake Photo — Training")
    print("=" * 55)

    # 1. Load data
    X, y = load_dataset(real_dir, screen_dir)
    print(f"\nDataset: {len(X)} total samples  ({sum(y==0)} real, {sum(y==1)} screen)")

    # 2. Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42
        ))
    ])

    # 3. Cross-validation (5-fold stratified)
    print("\nRunning 5-fold cross-validation ...")
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores  = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    f1s     = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')

    print(f"\n  CV Accuracy : {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
    print(f"  CV F1 Score : {f1s.mean():.3f} ± {f1s.std():.3f}")
    print(f"  Per-fold    : {[f'{s*100:.1f}%' for s in scores]}")

    # 4. Train on full dataset
    print("\nFitting final model on full dataset ...")
    pipeline.fit(X, y)

    # 5. Training-set report
    y_pred = pipeline.predict(X)
    print("\n── Full-dataset classification report ──")
    print(classification_report(y, y_pred, target_names=['Real', 'Screen']))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y, y_pred))

    # 6. Feature importance (top 10)
    feature_names = list(extract_features.__code__.co_consts)  # rough proxy
    importances   = pipeline.named_steps['clf'].feature_importances_
    top_idx       = np.argsort(importances)[::-1][:10]
    print("\n── Top 10 most important features ──")
    feat_keys = list(extract_features('/dev/null' if False else
        [p for p in glob.glob(os.path.join(real_dir, '*')) if os.path.isfile(p)][0]).keys())
    for rank, idx in enumerate(top_idx, 1):
        name = feat_keys[idx] if idx < len(feat_keys) else f"feat_{idx}"
        print(f"  {rank:2d}. {name:<25s}  importance={importances[idx]:.4f}")

    # 7. Save model
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")
    print("=" * 55)
    print("Done! Run predict.py to classify new images.")
    print("=" * 55)


if __name__ == '__main__':
    train(REAL_DIR, SCREEN_DIR, MODEL_PATH)