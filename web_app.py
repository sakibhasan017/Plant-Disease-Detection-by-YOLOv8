import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

@st.cache_resource
def load_disease_info():
    with open("disease_info.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
disease_info = load_disease_info()

st.markdown(
    """
    <style>
    :root{
      --bg:#0f1724;
      --card:#0b1220aa;
      --glass:#ffffff12;
      --accent1:#1dd3b0;
      --accent2:#4cc9f0;
      --muted:#cbd5e1;
    }
    html,body,header,section{background: linear-gradient(180deg, #071226 0%, #0f1724 100%) !important; color:var(--muted);}
    .app-header{
      display:flex;
      align-items:center;
      gap:18px;
      padding:18px 28px;
      border-radius:14px;
      background:linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      box-shadow: 0 6px 30px rgba(2,6,23,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
      margin-bottom:18px;
    }
    .logo{
      width:68px;
      height:68px;
      background:linear-gradient(135deg,var(--accent1),var(--accent2));
      display:flex;
      align-items:center;
      justify-content:center;
      border-radius:14px;
      box-shadow: 0 6px 18px rgba(12,18,22,0.6);
      font-size:28px;
      color:#012;
      font-weight:800;
    }
    .title{
      font-size:22px;
      font-weight:700;
      color:#e6f7f2;
      margin-bottom:4px;
    }
    .subtitle{
      font-size:13px;
      color:#9fb3be;
      margin-top:0;
    }
    .uploader{
      padding:18px;
      border-radius:12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.03);
      box-shadow: 0 8px 24px rgba(2,6,23,0.6);
    }
    .result-card{
      padding:16px;
      border-radius:12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid rgba(255,255,255,0.03);
      margin-bottom:14px;
    }
    .disease-name{
      font-size:18px;
      font-weight:700;
      color:#e6fff6;
    }
    .badge{
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      font-weight:700;
      font-size:12px;
      color:#05231f;
      background:linear-gradient(90deg,var(--accent1),var(--accent2));
      margin-left:8px;
    }
    .meta{
      color:#9fb3be;
      font-size:13px;
      margin-top:6px;
    }
    .cause, .cure{
      background: linear-gradient(180deg, rgba(255,255,255,0.012), rgba(255,255,255,0.006));
      padding:12px;
      border-radius:10px;
      border:1px solid rgba(255,255,255,0.02);
      margin-top:8px;
      color:#d7eef0;
    }
    .divider{
      height:1px;
      background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      margin:14px 0;
    }
    .small-muted{ color:#94a8b3; font-size:13px; }
    .count-box{
      background: rgba(255,255,255,0.02);
      padding:8px 12px;
      border-radius:10px;
      border:1px solid rgba(255,255,255,0.02);
      display:inline-block;
      color:#e6fff6;
      font-weight:700;
    }
    @media (max-width: 800px){
      .logo{width:56px;height:56px;font-size:24px}
      .title{font-size:18px}
    }
    </style>
    <div class="app-header">
      <div class="logo">🌿</div>
      <div>
        <div class="title">Plant Disease Detection</div>
        <div class="subtitle">Upload a photo of your plant leaves — instant diagnosis & treatment tips</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown('<div class="uploader">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Quick Settings", unsafe_allow_html=True)
    confidence_thresh = st.slider("Minimum confidence to show detection", 0.0, 1.0, 0.35, 0.01)
    show_boxes = st.checkbox("Show bounding boxes on image", value=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown("### About", unsafe_allow_html=True)
    st.markdown('<div class="small-muted">This app detects leaf diseases and gives cause & cure suggestions. Healthy leaves will show “No treatment required.”</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is None:
        st.image("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?q=80&w=1200&auto=format&fit=crop&ixlib=rb-4.0.3&s=5a8d8f7c2b8f1a6f5d1c2b8f5f7d9a0e", use_column_width=True)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        results = model.predict(img_array, imgsz=640)[0]
        boxes = results.boxes
        filtered = []
        for b in boxes:
            conf = float(b.conf)
            if conf >= confidence_thresh:
                filtered.append(b)
        if len(filtered) == 0:
            st.markdown('<div class="result-card"><div class="disease-name">No leaf detected</div><div class="meta">Please upload a clearer image showing a leaf close-up.</div></div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
        else:
            annotated = results.plot() if show_boxes else np.array(image)
            st.image(annotated, use_column_width=True)
            class_names = [results.names[int(b.cls)] for b in filtered]
            unique_classes = []
            for name in class_names:
                if name not in unique_classes:
                    unique_classes.append(name)
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="meta">Detected leaves: <span class="count-box">{len(class_names)}</span>  Unique diseases: <span class="count-box">{len(unique_classes)}</span></div>', unsafe_allow_html=True)
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            for cls in unique_classes:
                display_confidences = [float(b.conf) for b in filtered if results.names[int(b.cls)] == cls]
                best_conf = max(display_confidences) if display_confidences else 0.0
                conf_badge = f'<span class="badge">{best_conf:.2f}</span>'
                st.markdown(f'<div class="result-card"><div style="display:flex;align-items:center;justify-content:space-between;"><div class="disease-name">🌱 {cls}</div>{conf_badge}</div>', unsafe_allow_html=True)
                if cls in disease_info:
                    cause_text = disease_info[cls].get("cause", "Information not available.")
                    cure_text = disease_info[cls].get("cure", "Information not available.")
                    st.markdown(f'<div class="cause"><strong>Cause:</strong><div style="margin-top:6px">{cause_text}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="cure"><strong>Cure/Treatment:</strong><div style="margin-top:6px">{cure_text}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="cause">Information not available for this class.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
