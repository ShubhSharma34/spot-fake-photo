"""
app.py - Spot the Fake Photo
SalesCode AI Assignment — Streamlit Demo

Run: streamlit run app.py
"""

import os
import time
import warnings
import numpy as np
import joblib
import cv2
from scipy import stats
from skimage.feature import local_binary_pattern
import streamlit as st
from PIL import Image
import io

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CHANGE THIS to your actual folder path
BASE_DIR   = r"C:\Users\Shubh Sharma\Desktop\spot_fake"
MODEL_PATH = "model.pkl"
# ─────────────────────────────────────────────

# ── Page config ──────────────────────────────
st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Styling ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 720px; }

/* Header */
.app-header {
    text-align: center;
    margin-bottom: 2.5rem;
}
.app-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #0f0f0f;
    margin: 0;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-top: 0.4rem;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: #f0fdf4;
    color: #15803d;
    border: 1px solid #bbf7d0;
    border-radius: 99px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.6rem;
    letter-spacing: 0.3px;
}

/* Result cards */
.result-real {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-screen {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #fca5a5;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.result-label-real  { color: #15803d; }
.result-label-screen { color: #dc2626; }
.result-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    margin: 0.5rem 0 0 0;
}
.result-score-real   { color: #16a34a; }
.result-score-screen { color: #ef4444; }
.result-desc {
    font-size: 0.9rem;
    color: #6b7280;
    margin-top: 0.5rem;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.stat-box {
    flex: 1;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #111827;
}
.stat-label {
    font-size: 0.75rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}

/* Feature chips */
.features-section {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.features-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.8rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #374151;
    font-family: 'JetBrains Mono', monospace;
}
.chip-high { border-color: #ef4444; color: #dc2626; background: #fef2f2; }
.chip-low  { border-color: #22c55e; color: #15803d; background: #f0fdf4; }

/* Upload zone */
.upload-hint {
    text-align: center;
    color: #9ca3af;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* How it works */
.how-box {
    background: #fafafa;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 2rem;
}
.how-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.6rem;
}
.how-text {
    font-size: 0.85rem;
    color: #4b5563;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ── Feature extraction ────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def extract_features(img_array):
    """Extract features from a numpy image array (BGR)."""
    img  = cv2.resize(img_array, (512, 512))
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
    features['noise_std']      = float(noise.std())
    features['noise_kurtosis'] = float(stats.kurtosis(noise.flatten()))
    features['noise_mean_abs'] = float(np.abs(noise).mean())

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0,10), density=True)
    for i, v in enumerate(hist): features[f'lbp_hist_{i}'] = float(v)
    features['lbp_uniformity'] = float((lbp == lbp.max()).mean())

    for i, ch in enumerate(['b','g','r']):
        c = img[:,:,i].astype(float)
        features[f'{ch}_mean'] = float(c.mean())
        features[f'{ch}_std']  = float(c.std())
        features[f'{ch}_skew'] = float(stats.skew(c.flatten()))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features['saturation_mean'] = float(hsv[:,:,1].mean())
    features['saturation_std']  = float(hsv[:,:,1].std())
    features['value_mean']      = float(hsv[:,:,2].mean())
    features['value_std']       = float(hsv[:,:,2].std())

    edges = cv2.Laplacian(gray, cv2.CV_64F)
    features['laplacian_var']      = float(edges.var())
    features['laplacian_mean_abs'] = float(np.abs(edges).mean())
    features['laplacian_kurtosis'] = float(stats.kurtosis(edges.flatten()))

    h, w = gray.shape
    cv_ = float(gray[h//4:3*h//4, w//4:3*w//4].mean())
    corners = [gray[:h//8,:w//8].mean(), gray[:h//8,-w//8:].mean(),
               gray[-h//8:,:w//8].mean(), gray[-h//8:,-w//8:].mean()]
    cv2_ = float(np.mean(corners))
    features['vignette_ratio'] = float(cv_ / (cv2_ + 1e-6))
    features['vignette_diff']  = float(cv_ - cv2_)

    features['glare_ratio'] = float((gray > 240).mean())
    features['rb_ratio']    = float(img[:,:,2].mean() / (img[:,:,0].mean() + 1e-6))
    features['color_cast']  = float(img[:,:,2].astype(float).mean() - img[:,:,0].astype(float).mean())

    diff_h = np.abs(np.diff(gray.astype(float), axis=0))
    diff_v = np.abs(np.diff(gray.astype(float), axis=1))
    features['block_artifact_h'] = float(diff_h[7::8,:].mean() / (diff_h.mean() + 1e-6))
    features['block_artifact_v'] = float(diff_v[:,7::8].mean() / (diff_v.mean() + 1e-6))

    return features


def run_prediction(pil_image):
    """Run prediction on a PIL image. Returns (score, latency_ms, key_features)."""
    # Convert PIL → OpenCV BGR
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    model = load_model()

    t0    = time.perf_counter()
    feats = extract_features(img_bgr)
    feat_vec = np.array(list(feats.values())).reshape(1, -1)
    score = float(model.predict_proba(feat_vec)[0][1])
    latency = (time.perf_counter() - t0) * 1000

    key = {
        'noise_std':     round(feats['noise_std'], 2),
        'laplacian_var': round(feats['laplacian_var'], 1),
        'noise_kurtosis':round(feats['noise_kurtosis'], 1),
        'vignette_diff': round(feats['vignette_diff'], 2),
        'glare_ratio':   round(feats['glare_ratio'], 4),
    }
    return round(score, 4), round(latency, 1), key


# ── UI ───────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🔍 Spot the Fake Photo</p>
    <p class="app-subtitle">Detects whether an image is a real photo or a photo of a screen</p>
</div>
""", unsafe_allow_html=True)

# ── Input tabs ───────────────────────────────
tab1, tab2 = st.tabs([" Camera", " Upload Image"])

image_to_predict = None

with tab1:
    st.markdown('<p class="upload-hint">Take a photo directly — point at a real object OR at a screen showing a picture</p>', unsafe_allow_html=True)
    camera_image = st.camera_input("", label_visibility="collapsed")
    if camera_image:
        image_to_predict = Image.open(camera_image)

with tab2:
    st.markdown('<p class="upload-hint">Upload any photo — JPG, PNG, JPEG supported</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        image_to_predict = Image.open(uploaded)

# ── Prediction ───────────────────────────────
if image_to_predict:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image_to_predict, caption="Input image", use_container_width=True)

    with col2:
        with st.spinner("Analysing"):
            try:
                score, latency, key_feats = run_prediction(image_to_predict)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        is_screen = score > 0.5
        pct       = score * 100 if is_screen else (1 - score) * 100

        if is_screen:
            st.markdown(f"""
            <div class="result-screen">
                <p class="result-label result-label-screen">📺 SCREEN PHOTO</p>
                <p class="result-score result-score-screen">{score:.4f}</p>
                <p class="result-desc">This looks like a photo taken of a screen.<br>Confidence: {pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <p class="result-label result-label-real"> REAL PHOTO</p>
                <p class="result-score result-score-real">{score:.4f}</p>
                <p class="result-desc">This looks like a genuine real-world photo.<br>Confidence: {pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)


