"""
=================================================================
 Age Verification System — Streamlit Deployment
 Pipeline: YOLO Face Detection → CNN Age Prediction
           → Hardcoded Age Grouping → VLM Explainable Reasoning
 Aligned with Dissertation Methodology (Chapter 3)
=================================================================
"""

import os
import json
import time
import base64
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
import gdown

# Optional imports — graceful fallback
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ─── Configuration ────────────────────────────────────────────
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "age-verify-secret-2026")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
    MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
    IMAGE_SIZE = 224
    AGE_GROUPS = {"child": (0, 12), "teen": (13, 17), "adult": (18, 116)}
    GROUP_NAMES = ["child", "teen", "adult"]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEFAULT_BACKBONE = os.environ.get("MODEL_BACKBONE", "resnet50")

    # ── Google Drive Model Download ───────────────────────────
    # Replace with your Google Drive file ID
    # From: https://drive.google.com/file/d/YOUR_FILE_ID_HERE/view
    GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "1kH3U7bke9hOv3FJNADUd2bCAwjEnRvqz")
    MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "BEST_model.pth")

    # ── API Keys ──────────────────────────────────────────────
    @staticmethod
    def get_openai_key():
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            try:
                key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                key = ""
        return key


# ─── Model Definitions (must match training notebook exactly) ─
class AgePredictionCNN(nn.Module):
    """CNN for age regression + age group classification."""
    def __init__(self, backbone="resnet50", pretrained=False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            weights = "IMAGENET1K_V2" if pretrained else None
            base = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 2048
        elif backbone == "efficientnet_b0":
            weights = "IMAGENET1K_V1" if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            feat_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.age_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.group_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 3),
        )

    def forward(self, x):
        features = self.features(x)
        return self.age_head(features), self.group_head(features)


if HAS_TIMM:
    class ViTAgePrediction(nn.Module):
        """Vision Transformer for age prediction."""
        def __init__(self, model_name="vit_base_patch16_224", pretrained=False):
            super().__init__()
            self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            vit_dim = self.vit.num_features
            self.age_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(vit_dim, 512), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU(),
                nn.Linear(256, 1),
            )
            self.group_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(vit_dim, 256), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(256, 3),
            )

        def forward(self, x):
            features = self.vit(x)
            return self.age_head(features), self.group_head(features)


# ─── Pipeline Components ──────────────────────────────────────

def derive_age_group(predicted_age: float) -> str:
    """Hardcoded age grouping — Thesis §3.2.1"""
    if predicted_age < 13:
        return "child"
    elif predicted_age < 18:
        return "teen"
    return "adult"


def derive_decision(age_group: str) -> str:
    return "restrict" if age_group in ("child", "teen") else "allow"


class FaceDetector:
    """YOLO-based face detector with OpenCV Haar cascade fallback."""
    def __init__(self):
        self.yolo = None
        self.cascade = None

        if HAS_YOLO:
            try:
                self.yolo = YOLO("yolov8n.pt")
                logging.info("YOLO face detector loaded")
            except Exception as e:
                logging.warning(f"YOLO load failed: {e}")

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                self.cascade = None
                logging.warning("Haar cascade file failed to load")
            else:
                logging.info("Haar cascade loaded as backup")
        except Exception as e:
            logging.warning(f"Haar cascade load failed: {e}")
            self.cascade = None

    def _safe_crop(self, img, x1, y1, x2, y2, target_size):
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None, None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        return cv2.resize(crop, target_size), (x1, y1, x2, y2)

    def detect_and_crop(self, img_bgr, target_size=(224, 224)):
        if self.yolo is not None:
            try:
                results = self.yolo(img_bgr, verbose=False, conf=0.4)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    best = max(boxes, key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0]) * (b.xyxy[0][3]-b.xyxy[0][1])))
                    x1, y1, x2, y2 = [int(v) for v in best.xyxy[0].cpu().numpy()]
                    pad = int((x2 - x1) * 0.12)
                    face, box = self._safe_crop(img_bgr, x1 - pad, y1 - pad, x2 + pad, y2 + pad, target_size)
                    if face is not None:
                        return face, box
            except Exception as e:
                logging.debug(f"YOLO detection failed: {e}")

        if self.cascade is not None:
            try:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    pad = int(fw * 0.12)
                    face, box = self._safe_crop(img_bgr, x - pad, y - pad, x + fw + pad, y + fh + pad, target_size)
                    if face is not None:
                        return face, box
            except Exception as e:
                logging.debug(f"Haar detection failed: {e}")

        return cv2.resize(img_bgr, target_size), None


class VLMReasoner:
    """VLM explainable reasoning via OpenAI GPT-4 Vision."""

    PROMPT = """You are an age-verification AI for a research project on explainable age estimation.
Analyze this facial image carefully.

CNN MODEL PREDICTION (reference):
- Predicted Age: {cnn_age:.1f} years
- CNN Confidence: {cnn_confidence:.0%}
- CNN Age Group: {cnn_group}

YOUR TASK:
1. Independently estimate the person's age (0-100) from facial features
2. Classify: child (<13), teen (13-17), adult (18+)
3. List 3 visual cues that informed your estimate
4. State whether you agree with the CNN prediction

RESPOND WITH ONLY valid JSON (no markdown fences):
{{
  "final_predicted_age": <integer>,
  "final_age_group": "<child|teen|adult>",
  "reasoning": "<1-2 sentence explanation>",
  "visual_cues": ["<cue1>", "<cue2>", "<cue3>"],
  "cnn_input_used": {{ "cnn_predicted_age": {cnn_age_int}, "cnn_age_confidence": {cnn_confidence:.2f} }},
  "age_agreement": <true|false>,
  "age_discrepancy": <absolute diff between VLM and CNN age>,
  "confidence_level": <float 0.0-1.0>,
  "decision": "<allow|restrict>"
}}"""

    def __init__(self, api_key: str):
        self.client = None
        self.init_error = None
        if not HAS_OPENAI:
            self.init_error = "OpenAI SDK not installed"
            return
        if not api_key:
            self.init_error = "Missing API key"
            return
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            self.init_error = f"OpenAI client init failed: {e}"
            logging.warning(self.init_error)

    @property
    def available(self):
        return self.client is not None

    def predict(self, image_bytes: bytes, cnn_age: float, cnn_confidence: float, cnn_group: str):
        if not self.available:
            return None
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = self.PROMPT.format(
            cnn_age=cnn_age, cnn_confidence=cnn_confidence,
            cnn_group=cnn_group, cnn_age_int=int(round(cnn_age))
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return only valid JSON matching the schema."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}}
                    ]},
                ],
                max_tokens=500, temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            if "{" in raw and "}" in raw:
                raw = raw[raw.index("{"):raw.rindex("}") + 1]
            else:
                return {"error": "VLM response did not contain JSON"}
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logging.error(f"VLM JSON parse error: {e}")
            return {"error": f"Invalid JSON from VLM: {e}"}
        except Exception as e:
            logging.error(f"VLM error: {e}")
            return {"error": str(e)}


# ─── Google Drive Model Download ─────────────────────────────

def download_model_if_needed():
    """Download .pth from Google Drive if not already present."""
    model_dir = Path(Config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / Config.MODEL_FILENAME

    if model_path.exists():
        logging.info(f"Model already exists at {model_path}")
        return model_path

    file_id = Config.GDRIVE_FILE_ID
    if file_id == "YOUR_FILE_ID_HERE":
        logging.warning("No Google Drive file ID configured — skipping model download")
        return None

    url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Downloading model from Google Drive: {url}")

    try:
        gdown.download(url, str(model_path), quiet=False)
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logging.info(f"Model downloaded successfully: {model_path} ({size_mb:.1f} MB)")
            return model_path
        else:
            logging.error("Download appeared to succeed but file not found")
            return None
    except Exception as e:
        logging.error(f"Model download failed: {e}")
        return None


# ─── Load Models (cached via st.cache_resource) ──────────────

@st.cache_resource
def load_pipeline(openai_api_key: str | None = None):
    """Load all ML components once and cache them across reruns."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Download model from Google Drive if not cached locally
    download_model_if_needed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    face_detector = FaceDetector()

    cnn_model = None
    model_meta = {}
    backbone = Config.DEFAULT_BACKBONE

    # Try loading saved checkpoint from MODEL_DIR
    model_files = list(Path(Config.MODEL_DIR).glob("*.pth"))
    if model_files:
        best_files = [f for f in model_files if "BEST" in f.name.upper()]
        ckpt_path = best_files[0] if best_files else model_files[0]
        logging.info(f"Loading checkpoint: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            saved_name = checkpoint.get("model_name", "")
            if "efficientnet" in saved_name.lower():
                backbone = "efficientnet_b0"
            elif "vit" in saved_name.lower() and HAS_TIMM:
                backbone = "vit_base"
            else:
                backbone = "resnet50"

            if backbone == "vit_base" and HAS_TIMM:
                cnn_model = ViTAgePrediction(pretrained=False)
            else:
                cnn_model = AgePredictionCNN(backbone=backbone, pretrained=False)

            cnn_model.load_state_dict(checkpoint["model_state_dict"])
            cnn_model.to(device).eval()
            model_meta = checkpoint.get("metrics", {})
            logging.info(f"Model loaded: {saved_name} | MAE={model_meta.get('mae', '?')}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            cnn_model = None

    # If no checkpoint, load pretrained backbone as demo
    if cnn_model is None:
        logging.info("No checkpoint found — loading pretrained ResNet-50 (demo mode)")
        cnn_model = AgePredictionCNN(backbone="resnet50", pretrained=True)
        cnn_model.to(device).eval()
        model_meta = {"note": "Demo mode — pretrained on ImageNet, not fine-tuned on age data. Upload your .pth to /models for real predictions."}

    key = openai_api_key or Config.get_openai_key()
    vlm = VLMReasoner(key)
    logging.info(f"VLM available: {vlm.available}")

    return device, face_detector, cnn_model, backbone, model_meta, vlm


def preprocess_image(img_bgr, device):
    """Convert BGR image to normalised tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for c in range(3):
        img_rgb[:, :, c] = (img_rgb[:, :, c] - Config.IMAGENET_MEAN[c]) / Config.IMAGENET_STD[c]
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor.to(device)


def image_to_jpeg_bytes(img_bgr):
    success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return buf.tobytes()


def run_pipeline(img_bgr, use_vlm, device, face_detector, cnn_model, backbone, vlm):
    """Run the full age-verification pipeline. Returns a results dict."""
    start = time.time()

    # Step 1: Face Detection
    t_face = time.time()
    face_crop, face_box = face_detector.detect_and_crop(img_bgr, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
    face_detected = face_box is not None
    face_time = time.time() - t_face

    # Step 2: CNN Age Prediction
    t_cnn = time.time()
    tensor = preprocess_image(face_crop, device)
    with torch.no_grad():
        age_pred, group_pred = cnn_model(tensor)

    cnn_age = float(age_pred.item())
    cnn_age = max(0.0, min(100.0, cnn_age))
    group_probs = torch.softmax(group_pred, dim=1).cpu().numpy()[0]
    cnn_confidence = float(group_probs.max())
    cnn_time = time.time() - t_cnn

    # Step 3: Hardcoded Age Grouping
    cnn_group = derive_age_group(cnn_age)
    cnn_decision = derive_decision(cnn_group)

    # Step 4: VLM Reasoning (optional)
    vlm_result = None
    vlm_time = 0
    if use_vlm and vlm.available:
        t_vlm = time.time()
        try:
            jpeg_bytes = image_to_jpeg_bytes(face_crop)
            vlm_result = vlm.predict(jpeg_bytes, cnn_age, cnn_confidence, cnn_group)
        except Exception as e:
            vlm_result = {"error": str(e)}
        vlm_time = time.time() - t_vlm

    # Annotated image
    annotated = img_bgr.copy()
    if face_box:
        x1, y1, x2, y2 = face_box
        color = (0, 200, 0) if cnn_group == "adult" else (0, 165, 255) if cnn_group == "teen" else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        label = f"Age:{cnn_age:.0f} ({cnn_group})"
        text_y = y1 - 10 if y1 > 30 else y2 + 25
        cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    total_time = time.time() - start
    has_vlm = vlm_result is not None and "error" not in vlm_result

    return {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "annotated_bgr": annotated,
        "face_crop_bgr": face_crop,
        "pipeline": {
            "face_detection": {
                "method": "YOLO" if (face_detector.yolo and face_detected) else "Haar Cascade" if face_detected else "Fallback (full image)",
                "face_detected": face_detected,
                "bounding_box": list(face_box) if face_box else None,
            },
            "cnn_prediction": {
                "model": backbone,
                "cnn_predicted_age": round(cnn_age, 1),
                "cnn_predicted_age_int": int(round(cnn_age)),
                "cnn_age_confidence": round(cnn_confidence, 3),
                "group_probabilities": {
                    "child": round(float(group_probs[0]), 3),
                    "teen": round(float(group_probs[1]), 3),
                    "adult": round(float(group_probs[2]), 3),
                },
            },
            "age_grouping": {
                "method": "hardcoded_threshold",
                "rules": "child < 13 | teen 13-17 | adult >= 18",
                "cnn_age_group": cnn_group,
                "cnn_decision": cnn_decision,
            },
            "vlm_reasoning": vlm_result if vlm_result else {
                "status": "skipped" if not use_vlm else "unavailable",
                "note": "Enable VLM for explainable reasoning"
            },
        },
        "final_decision": {
            "predicted_age": vlm_result.get("final_predicted_age", int(round(cnn_age))) if has_vlm else int(round(cnn_age)),
            "age_group": vlm_result.get("final_age_group", cnn_group) if has_vlm else cnn_group,
            "decision": vlm_result.get("decision", cnn_decision) if has_vlm else cnn_decision,
            "source": "VLM (GPT-4V)" if has_vlm else "CNN + Hardcoded Rules",
        },
        "latency": {
            "face_detection_ms": round(face_time * 1000, 1),
            "cnn_inference_ms": round(cnn_time * 1000, 1),
            "vlm_reasoning_ms": round(vlm_time * 1000, 1) if vlm_time else None,
            "total_ms": round(total_time * 1000, 1),
        },
    }


# ─── Custom CSS ───────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --green: #22c55e;
        --red: #ef4444;
        --amber: #f59e0b;
        --accent: #3b82f6;
    }

    .stApp { font-family: 'DM Sans', sans-serif; }

    .decision-banner {
        padding: 20px 24px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
    }
    .decision-banner.allow {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.05));
        border: 1px solid rgba(34,197,94,0.35);
    }
    .decision-banner.restrict {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.35);
    }
    .decision-banner .icon { font-size: 32px; }
    .decision-banner h3 { margin: 0; }
    .decision-banner.allow h3 { color: var(--green); }
    .decision-banner.restrict h3 { color: var(--red); }
    .decision-banner p { margin: 4px 0 0; font-size: 14px; opacity: 0.8; }

    .pipeline-step {
        background: rgba(26, 34, 53, 0.6);
        border: 1px solid rgba(30, 41, 59, 0.8);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    .pipeline-step-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .pipeline-step-header .num {
        width: 24px; height: 24px; border-radius: 6px;
        font-size: 12px; font-weight: 700; color: #fff;
        display: inline-flex; align-items: center; justify-content: center;
    }
    .num-face { background: #8b5cf6; }
    .num-cnn { background: #3b82f6; }
    .num-group { background: #f59e0b; }
    .num-vlm { background: #22c55e; }
    .pipeline-step-header h4 { margin: 0; font-size: 14px; font-weight: 600; }
    .pipeline-step-header .latency {
        margin-left: auto; font-size: 12px;
        font-family: 'JetBrains Mono', monospace; opacity: 0.6;
    }

    .prob-bar-container { margin-top: 8px; }
    .prob-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
    .prob-bar .lbl { font-size: 12px; width: 48px; font-weight: 600; opacity: 0.7; }
    .prob-bar .track {
        flex: 1; height: 8px; background: rgba(255,255,255,0.06);
        border-radius: 4px; overflow: hidden;
    }
    .prob-bar .fill { height: 100%; border-radius: 4px; }
    .prob-bar .val { font-size: 12px; font-family: 'JetBrains Mono', monospace; width: 48px; text-align: right; opacity: 0.7; }

    .vlm-cue {
        display: inline-block;
        font-size: 12px; padding: 4px 10px;
        background: rgba(34,197,94,0.1); color: var(--green);
        border-radius: 20px; border: 1px solid rgba(34,197,94,0.25);
        margin: 3px 3px 3px 0;
    }
    .vlm-reasoning-text {
        font-size: 14px; line-height: 1.6;
        padding: 10px 14px; background: rgba(0,0,0,0.15);
        border-radius: 6px; margin-top: 8px;
        border-left: 3px solid var(--green);
    }

    div[data-testid="stMetric"] {
        background: rgba(0,0,0,0.15);
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
"""


# ─── Streamlit App ────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Age Verification System — Hybrid CNN + VLM",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # API key input (session only)
    with st.expander("🔑 API Key", expanded=False):
        stored_key = st.session_state.get("openai_api_key", "")
        key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=stored_key,
            placeholder="sk-...",
            help="Stored only for this session and not written to disk.",
        )
        if key_input != stored_key:
            st.session_state["openai_api_key"] = key_input
            st.cache_resource.clear()
            st.rerun()
        if st.button("Clear key"):
            st.session_state["openai_api_key"] = ""
            st.cache_resource.clear()
            st.rerun()

    # Load pipeline (cached)
    session_key = st.session_state.get("openai_api_key", "")
    device, face_detector, cnn_model, backbone, model_meta, vlm = load_pipeline(
        session_key if session_key else None
    )

    # ── Header ────────────────────────────────────────────────
    col_title, col_badges = st.columns([3, 2])
    with col_title:
        st.markdown("# 🛡️ Age Verification System")
        st.caption("YOLO → CNN → Age Grouping → VLM Reasoning")
    with col_badges:
        bc1, bc2, bc3 = st.columns(3)
        bc1.success(f"**{backbone}**")
        bc2.success(f"**{device}**")
        if vlm.available:
            bc3.success("**VLM ON**")
        else:
            bc3.warning("**VLM OFF**")

    st.divider()

    # ── Layout: Input (left) | Results (right) ────────────────
    col_input, col_results = st.columns([1, 2], gap="large")

    with col_input:
        st.subheader("📥 Input")

        input_mode = st.radio(
            "Input method",
            ["📁 Upload", "📷 Webcam"],
            horizontal=True,
            label_visibility="collapsed",
        )

        img_bgr = None

        if input_mode == "📁 Upload":
            uploaded = st.file_uploader(
                "Upload an image",
                type=["png", "jpg", "jpeg", "gif", "webp", "bmp"],
                label_visibility="collapsed",
            )
            if uploaded is not None:
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)
                else:
                    st.error("Failed to decode image.")

        else:  # Webcam
            cam_image = st.camera_input("Capture from webcam", label_visibility="collapsed")
            if cam_image is not None:
                file_bytes = np.frombuffer(cam_image.getvalue(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.markdown("---")
        use_vlm = st.toggle(
            "🧠 VLM Reasoning (GPT-4V)",
            value=vlm.available,
            help="Enable GPT-4V for explainable age reasoning",
        )
        if not vlm.available:
            reason = vlm.init_error or "VLM unavailable"
            st.warning(f"VLM unavailable: {reason}")

        run_btn = st.button(
            "🔬 Run Age Verification",
            type="primary",
            use_container_width=True,
            disabled=img_bgr is None,
        )

        # Pipeline legend
        with st.expander("ℹ️ Pipeline Steps", expanded=False):
            st.markdown("""
            **❶** YOLO Face Detection & Crop  
            **❷** CNN Numerical Age Prediction  
            **❸** Hardcoded Age Group (child/teen/adult)  
            **❹** VLM Explainable Reasoning (optional)
            """)

        if "note" in model_meta:
            st.info(model_meta["note"], icon="⚠️")

    # ── Results ───────────────────────────────────────────────
    with col_results:
        st.subheader("📊 Results")

        if run_btn and img_bgr is not None:
            with st.spinner("Running CNN" + (" + VLM" if use_vlm else "") + " pipeline…"):
                data = run_pipeline(img_bgr, use_vlm, device, face_detector, cnn_model, backbone, vlm)

            if not data["success"]:
                st.error("Pipeline failed.")
                return

            d = data["final_decision"]
            p = data["pipeline"]
            lat = data["latency"]
            is_allow = d["decision"] == "allow"

            # Decision banner
            banner_cls = "allow" if is_allow else "restrict"
            icon = "✅" if is_allow else "🚫"
            title = "ACCESS ALLOWED" if is_allow else "ACCESS RESTRICTED"
            st.markdown(f"""
            <div class="decision-banner {banner_cls}">
                <span class="icon">{icon}</span>
                <div>
                    <h3>{title}</h3>
                    <p>Predicted age: <strong>{d['predicted_age']}</strong> years — classified as
                    <strong>{d['age_group'].upper()}</strong> — via {d['source']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Images
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.caption("ANNOTATED IMAGE")
                st.image(cv2.cvtColor(data["annotated_bgr"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with img_col2:
                st.caption("DETECTED FACE CROP")
                st.image(cv2.cvtColor(data["face_crop_bgr"], cv2.COLOR_BGR2RGB), use_container_width=True)

            # Step 1: Face Detection
            fd = p["face_detection"]
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-step-header">
                    <span class="num num-face">1</span>
                    <h4>Face Detection</h4>
                    <span class="latency">{lat['face_detection_ms']}ms</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            fc1.metric("Method", fd["method"])
            fc2.metric("Detected", "✅ Yes" if fd["face_detected"] else "⚠️ No")

            # Step 2: CNN Prediction
            cnn = p["cnn_prediction"]
            gp = cnn["group_probabilities"]
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-step-header">
                    <span class="num num-cnn">2</span>
                    <h4>CNN Age Prediction</h4>
                    <span class="latency">{lat['cnn_inference_ms']}ms</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Predicted Age", f"{cnn['cnn_predicted_age']}")
            mc2.metric("Confidence", f"{cnn['cnn_age_confidence'] * 100:.1f}%")
            mc3.metric("Model", cnn["model"])

            # Probability bars
            colors = {"child": "#ef4444", "teen": "#f59e0b", "adult": "#22c55e"}
            bars_html = '<div class="prob-bar-container">'
            for g in ["child", "teen", "adult"]:
                pct = gp[g] * 100
                bars_html += f"""
                <div class="prob-bar">
                    <span class="lbl">{g}</span>
                    <div class="track"><div class="fill" style="width:{pct:.1f}%;background:{colors[g]};"></div></div>
                    <span class="val">{pct:.1f}%</span>
                </div>"""
            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)

            # Step 3: Age Grouping
            ag = p["age_grouping"]
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-step-header">
                    <span class="num num-group">3</span>
                    <h4>Hardcoded Age Grouping</h4>
                    <span class="latency">&lt;1ms</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            gc1, gc2, gc3 = st.columns(3)
            gc1.metric("Age Group", ag["cnn_age_group"].upper())
            gc2.metric("Decision", ag["cnn_decision"].upper())
            gc3.metric("Rules", ag["rules"])

            # Step 4: VLM Reasoning
            vlm_data = p["vlm_reasoning"]
            has_vlm_result = vlm_data and vlm_data.get("final_predicted_age") is not None and "error" not in vlm_data
            vlm_latency_str = f"{lat['vlm_reasoning_ms']}ms" if lat["vlm_reasoning_ms"] else "skipped"

            st.markdown(f"""
            <div class="pipeline-step">
                <div class="pipeline-step-header">
                    <span class="num num-vlm">4</span>
                    <h4>VLM Explainable Reasoning</h4>
                    <span class="latency">{vlm_latency_str}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if has_vlm_result:
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.metric("VLM Age", vlm_data["final_predicted_age"])
                vc2.metric("VLM Group", vlm_data["final_age_group"].upper())
                vc3.metric("Agreement", "✅" if vlm_data.get("age_agreement") else "⚠️")
                vc4.metric("Discrepancy", f"{vlm_data.get('age_discrepancy', 0)} yrs")

                if vlm_data.get("reasoning"):
                    st.markdown(f'<div class="vlm-reasoning-text">{vlm_data["reasoning"]}</div>', unsafe_allow_html=True)
                if vlm_data.get("visual_cues"):
                    cues_html = "".join(f'<span class="vlm-cue">{c}</span>' for c in vlm_data["visual_cues"])
                    st.markdown(cues_html, unsafe_allow_html=True)
            else:
                note = vlm_data.get("note") or vlm_data.get("status") or "VLM not enabled — toggle it on for explainable reasoning"
                st.caption(note)

            # Latency summary
            st.markdown("---")
            lc = st.columns(4 if lat["vlm_reasoning_ms"] else 3)
            lc[0].metric("Face Detect", f"{lat['face_detection_ms']}ms")
            lc[1].metric("CNN Infer", f"{lat['cnn_inference_ms']}ms")
            if lat["vlm_reasoning_ms"]:
                lc[2].metric("VLM", f"{lat['vlm_reasoning_ms']}ms")
            lc[-1].metric("Total", f"{lat['total_ms']}ms")

            # JSON viewer
            with st.expander("▸ View Full JSON Response"):
                display_data = {k: v for k, v in data.items() if k not in ("annotated_bgr", "face_crop_bgr")}
                st.json(display_data)

        else:
            st.markdown("""
            <div style="text-align:center; padding:80px 20px; opacity:0.4;">
                <div style="font-size:56px;">🔍</div>
                <h3>Upload an image to begin</h3>
                <p>The system will detect faces, predict age, and provide explainable reasoning</p>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.divider()
    fc1, fc2 = st.columns(2)
    fc1.caption("Age Verification System — Dissertation Research Project")
    fc2.caption("Pipeline: YOLO + CNN + VLM | Hybrid Architecture")


if __name__ == "__main__":
    main()