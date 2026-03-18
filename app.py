
# ============================================================
#  FLOODXPLAIN - Streamlit Dashboard
#  Run with: streamlit run app.py
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──
st.set_page_config(
    page_title="FloodXPlain",
    page_icon="🌊",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main { background-color: #0A0F1E; }
    .stApp { background-color: #0A0F1E; }
    h1 { color: #FFFFFF !important; }
    h2, h3 { color: #A0B0FF !important; }
    p, label { color: #C0C8E8 !important; }
    .result-box {
        background: #1A2456;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3050A0;
        text-align: center;
    }
    .metric-box {
        background: #0D1830;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2040A0;
        margin: 5px 0;
    }
    .flood-result {
        color: #FF4444 !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    .nonflood-result {
        color: #44FF88 !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    .header-box {
        background: linear-gradient(135deg, #0D1830, #1A2456);
        border-radius: 16px;
        padding: 30px;
        border: 1px solid #3050A0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="header-box">
    <h1 style="font-size:36px; margin:0">🌊 FloodXPlain</h1>
    <p style="font-size:16px; margin:8px 0 0 0; color:#A0B0FF">
        XAI-Driven CNN Framework for Post-Flood Impact Analysis
    </p>
    <p style="font-size:12px; margin:4px 0 0 0; color:#6070A0">
        ResNet-50 · GradCAM · GradCAM++ · 99.96% Accuracy
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load model ──
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Load your trained weights
    model.load_state_dict(
        torch.load('floodxplain_model.pth', map_location='cpu')
    )
    model.eval()
    return model

# ── Transform ──
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = ['Flood Images', 'Non Flood Images']
CLASS_LABELS = ['🔴  FLOODED', '🟢  NOT FLOODED']
CLASS_COLORS = ['#FF4444', '#44FF88']

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ XAI Settings")
    xai_method = st.radio(
        "Select XAI Method:",
        ["GradCAM", "GradCAM++", "Both Side by Side"],
        index=2
    )
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    <div class="metric-box">
        <p style="margin:0; font-size:13px">🧠 Model: ResNet-50</p>
        <p style="margin:0; font-size:13px">📈 Val Accuracy: 99.96%</p>
        <p style="margin:0; font-size:13px">🗂️ Dataset: 13,044 images</p>
        <p style="margin:0; font-size:13px">⚡ Classes: Flood / Non-Flood</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 👥 Team")
    st.markdown("""
    <div class="metric-box">
        <p style="margin:0; font-size:12px">Satyam Ponnada · 22BCE9025</p>
        <p style="margin:0; font-size:12px">Phani Alapati · 22BCE7293</p>
        <p style="margin:0; font-size:12px">Mihir Datta · 22BCE7636</p>
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──
st.markdown("### 📤 Upload Aerial Image")
uploaded_file = st.file_uploader(
    "Upload a flood or non-flood aerial image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Load model
        with st.spinner("🧠 Loading FloodXPlain model..."):
            model = load_model()

        # Load and process image
        image = Image.open(uploaded_file).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            pred_idx = output.argmax(1).item()
            confidence = probabilities[pred_idx].item() * 100
            flood_prob = probabilities[0].item() * 100
            nonflood_prob = probabilities[1].item() * 100

        # Convert image for display
        img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        img_np = np.clip(img_np, 0, 1)

        # Generate heatmaps
        target_layer = [model.layer4[-1]]
        with st.spinner("🔥 Generating XAI heatmaps..."):
            gradcam_obj = GradCAM(
                model=model,
                target_layers=target_layer
            )
            gradcam_plus_obj = GradCAMPlusPlus(
                model=model,
                target_layers=target_layer
            )
            cam1 = gradcam_obj(input_tensor=img_tensor)
            cam2 = gradcam_plus_obj(input_tensor=img_tensor)
            overlay1 = show_cam_on_image(img_np, cam1[0], use_rgb=True)
            overlay2 = show_cam_on_image(img_np, cam2[0], use_rgb=True)

        st.markdown("---")

        # ── Results Row ──
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.markdown("#### 🛰️ Input Image")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Prediction Result")
            label_text = CLASS_LABELS[pred_idx]
            color = CLASS_COLORS[pred_idx]
            st.markdown(f"""
            <div class="result-box">
                <p style="color:{color}; font-size:26px;
                   font-weight:bold; margin:0">{label_text}</p>
                <p style="color:#A0B0FF; font-size:18px;
                   margin:8px 0">Confidence: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence bars
            st.markdown("**Class Probabilities:**")
            st.markdown(f"🔴 Flooded")
            st.progress(int(flood_prob))
            st.caption(f"{flood_prob:.2f}%")

            st.markdown(f"🟢 Not Flooded")
            st.progress(int(nonflood_prob))
            st.caption(f"{nonflood_prob:.2f}%")

        with col3:
            st.markdown("#### 📋 Damage Assessment")
            if pred_idx == 0:
                severity = "HIGH" if confidence > 90 else "MEDIUM"
                sev_color = "#FF4444" if severity == "HIGH" else "#FF8800"
                st.markdown(f"""
                <div class="metric-box">
                    <p style="color:{sev_color}; font-size:20px;
                       font-weight:bold; margin:0">
                       ⚠️ Severity: {severity}</p>
                    <p style="font-size:12px; margin:8px 0 0 0">
                       Flood detected in this aerial image.<br>
                       Immediate response recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-box">
                    <p style="color:#44FF88; font-size:20px;
                       font-weight:bold; margin:0">
                       ✅ Area: SAFE</p>
                    <p style="font-size:12px; margin:8px 0 0 0">
                       No flood detected in this area.<br>
                       No immediate action needed.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-box" style="margin-top:10px">
                <p style="font-size:12px; margin:0">
                🧠 Model: ResNet-50<br>
                📊 XAI: {xai_method}<br>
                🎯 Confidence: {confidence:.2f}%<br>
                📐 Input size: 224×224
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── XAI Heatmaps ──
        st.markdown("---")
        st.markdown("### 🔥 XAI Explainability Heatmaps")
        st.caption(
            "Red/warm areas = regions the AI focused on to make its decision. "
            "This is what makes FloodXPlain trustworthy."
        )

        if xai_method == "GradCAM":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(img_np, use_container_width=True)
            with col2:
                st.markdown("**GradCAM Heatmap**")
                st.image(overlay1, use_container_width=True)

        elif xai_method == "GradCAM++":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Image**")
                st.image(img_np, use_container_width=True)
            with col2:
                st.markdown("**GradCAM++ Heatmap**")
                st.image(overlay2, use_container_width=True)

        else:  # Both
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Original Image**")
                st.image(img_np, use_container_width=True)
            with col2:
                st.markdown("**GradCAM**")
                st.image(overlay1, use_container_width=True)
            with col3:
                st.markdown("**GradCAM++**")
                st.image(overlay2, use_container_width=True)

        st.markdown("""
        <div class="metric-box" style="margin-top:10px">
            <p style="font-size:12px; margin:0; color:#A0B0FF">
            💡 <b>How to read the heatmap:</b>
            Red = very high attention (AI strongly focused here) ·
            Yellow = medium attention ·
            Blue = low attention (AI ignored this region).
            For flood images, the AI should focus on water surface areas.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.success(
            f"✅ Analysis complete! "
            f"Prediction: {CLASS_NAMES[pred_idx]} "
            f"with {confidence:.2f}% confidence."
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info(
            "Make sure floodxplain_model.pth is in the "
            "same folder as app.py"
        )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding:40px">
        <p style="font-size:60px">🌊</p>
        <p style="font-size:20px; color:#A0B0FF">
            Upload an aerial image to begin flood analysis
        </p>
        <p style="font-size:14px; color:#6070A0">
            Supports JPG and PNG aerial/satellite images
        </p>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("---")
    st.markdown("### How FloodXPlain Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box" style="text-align:center">
            <p style="font-size:30px; margin:0">📤</p>
            <p style="font-weight:bold; margin:4px 0">Upload</p>
            <p style="font-size:12px">Upload aerial image from drone or satellite</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box" style="text-align:center">
            <p style="font-size:30px; margin:0">🧠</p>
            <p style="font-weight:bold; margin:4px 0">Analyse</p>
            <p style="font-size:12px">ResNet-50 CNN classifies flood damage</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box" style="text-align:center">
            <p style="font-size:30px; margin:0">🔥</p>
            <p style="font-weight:bold; margin:4px 0">Explain</p>
            <p style="font-size:12px">GradCAM shows why the AI decided</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box" style="text-align:center">
            <p style="font-size:30px; margin:0">🎯</p>
            <p style="font-weight:bold; margin:4px 0">Decide</p>
            <p style="font-size:12px">Disaster managers get trusted results</p>
        </div>
        """, unsafe_allow_html=True)
