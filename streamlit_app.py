"""
=================================================================
 Age Verification System v2.0 — Streamlit Deployment
 Pipeline: YOLO Multi-Face Detection → CNN Age Prediction
           → Hardcoded Age Grouping → VLM Per-Face Reasoning
 Supports: Image upload, Webcam, Video upload, Multi-face
=================================================================
"""

import os
import re
import json
import time
import base64
import logging
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"

# ── Headless-server shared-library guard ────────────────────────────────
# Streamlit Cloud (Debian trixie) may be missing libs that the full
# opencv-python build (forced by ultralytics) needs at import time.
# We stub every one of them before `import cv2` runs.
def _ensure_libgl():
    import ctypes, ctypes.util, subprocess

    NEEDED = [
        "libGL.so.1",
        "libgthread-2.0.so.0",
        "libglib-2.0.so.0",
        "libGLdispatch.so.0",
    ]

    def _make_stub(name: str) -> str:
        path = os.path.join(tempfile.gettempdir(), name)
        if os.path.exists(path):
            return path
        # Try gcc first (available on most CI images)
        try:
            r = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-nostdlib", "-o", path,
                 "-x", "c", "/dev/null"],
                capture_output=True, timeout=15
            )
            if r.returncode == 0 and os.path.exists(path):
                return path
        except Exception:
            pass
        # Fallback: hand-craft a minimal 64-bit ELF DSO (344 bytes)
        import struct
        e_ident = (b"\x7fELF" + b"\x02\x01\x01" + b"\x00" * 9)
        header = struct.pack(
            "<HHIQQQIHHHHHH",
            3, 0x3E, 1,   # ET_DYN, x86-64, version
            0, 0, 0,      # entry, phoff, shoff
            0, 64, 0, 0,  # flags, ehsize, phentsize, phnum
            64, 0, 0      # shentsize, shnum, shstrndx
        )
        with open(path, "wb") as fh:
            fh.write(e_ident + header)
        return path

    for libname in NEEDED:
        if ctypes.util.find_library(libname.split(".")[0].replace("lib", "", 1)):
            continue  # already on system
        try:
            stub = _make_stub(libname)
            ctypes.CDLL(stub)
            prev = os.environ.get("LD_PRELOAD", "")
            if stub not in prev:
                os.environ["LD_PRELOAD"] = f"{stub}:{prev}".rstrip(":")
        except OSError:
            pass

_ensure_libgl()

import cv2
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
import requests

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

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

import albumentations as A
from albumentations.pytorch import ToTensorV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Configuration ─────────────────────────────────────────────────────────
class Config:
    MODEL_DIR = os.environ.get(
        "MODEL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    )
    IMAGE_SIZE = 224
    AGE_GROUPS = {"child": (0, 12), "teen": (13, 17), "adult": (18, 116)}
    GROUP_NAMES = ["child", "teen", "adult"]
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # ── Backbones supported ──────────────────────────────────────────────
    # Set MODEL_BACKBONE to match the backbone used during training
    DEFAULT_BACKBONE = os.environ.get("MODEL_BACKBONE", "densenet")

    CNN_FILENAME    = os.environ.get("CNN_FILENAME",  "densenet_best.pth")
    YOLO_FILENAME   = os.environ.get("YOLO_FILENAME", "yolo_face_best.pt")

    BACKBONE_FILES = {
        "densenet":        "densenet_best.pth",
        "resnet50":        "resnet50_best.pth",
        "efficientnet_v2": "efficientnet_v2_best.pth",
        "convnext":        "convnext_best.pth",
        "vit-base":        "vit_base_best.pth",
        "vit-tiny":        "vit_base_best.pth",  # no separate tiny weights; falls back to base
    }

    # ── GitHub Release download URLs (no auth required for public repo) ──
    _GH_BASE = "https://github.com/Mystique1337/age_prediction_hybrid-/releases/download/v1.0-models"
    MODEL_URLS = {
        "densenet_best.pth":        f"{_GH_BASE}/densenet_best.pth",
        "resnet50_best.pth":        f"{_GH_BASE}/resnet50_best.pth",
        "efficientnet_v2_best.pth": f"{_GH_BASE}/efficientnet_v2_best.pth",
        "convnext_best.pth":        f"{_GH_BASE}/convnext_best.pth",
        "vit_base_best.pth":        f"{_GH_BASE}/vit_base_best.pth",
        "yolo_face_best.pt":        f"{_GH_BASE}/yolo_face_best.pt",
    }

    @staticmethod
    def get_openai_key():
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            try:
                key = st.secrets.get("OPENAI_API_KEY", "")
            except Exception:
                key = ""
        return key


# ─── Model Definitions (must match training architecture) ──────────────────
class AgePredictionCNN(nn.Module):
    """Mirrors the training-time AgePredictionCNN. Supports all trained backbones."""

    def __init__(self, backbone: str = "densenet", pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet50":
            base = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            self.features = nn.Sequential(*list(base.children())[:-1])
            feat_dim = 2048
        elif backbone == "efficientnet_v2":
            base = models.efficientnet_v2_m(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
            feat_dim = 1280
        elif backbone == "convnext":
            base = models.convnext_base(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
            feat_dim = 1024
        elif backbone == "densenet":
            base = models.densenet201(weights="IMAGENET1K_V1" if pretrained else None)
            self.features = base.features
            feat_dim = 1920
        else:
            raise ValueError(f"Unsupported CNN backbone: {backbone}")

        self.age_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.group_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        feats = self.features(x)
        if isinstance(feats, torch.Tensor) and feats.dim() == 2:
            feats = feats.unsqueeze(-1).unsqueeze(-1)
        return self.age_head(feats), self.group_head(feats)


class ViTAgePrediction(nn.Module):
    """ViT model — used when backbone is 'vit-base' or 'vit-tiny'."""

    def __init__(self, model_name: str = "vit_base_patch16_224"):
        super().__init__()
        if not HAS_TIMM:
            raise ImportError("timm is required for ViT models. pip install timm")
        self.vit = timm.create_model(model_name, pretrained=False, num_classes=0)
        vit_dim = self.vit.num_features

        self.age_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(vit_dim, 512), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 1),
        )
        self.group_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(vit_dim, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        feats = self.vit(x)
        return self.age_head(feats), self.group_head(feats)


# ─── Model Download Helpers ────────────────────────────────────────────────
def _is_valid_model_file(path: str, min_bytes: int = 1 << 20) -> bool:
    """Return True only if path is a non-empty PyTorch zip archive (magic PK\x03\x04)."""
    try:
        if os.path.getsize(path) < min_bytes:
            return False
        with open(path, "rb") as fh:
            return fh.read(4) == b"PK\x03\x04"
    except Exception:
        return False


def _gh_download(url: str, dest_path: str, label: str) -> bool:
    """Download a file from a GitHub Release URL with progress display."""
    if os.path.exists(dest_path):
        if _is_valid_model_file(dest_path):
            return True
        logger.warning(f"Corrupt/incomplete file detected at {dest_path}; re-downloading.")
        os.remove(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        st.info(f"⬇️  Downloading {label} …")
        with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            bar = st.progress(0, text=f"Downloading {label}…")
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        bar.progress(min(downloaded / total, 1.0), text=f"Downloading {label}… {downloaded>>20}MB/{total>>20}MB")
            bar.empty()
        return os.path.exists(dest_path)
    except Exception as e:
        logger.warning(f"GitHub download failed for {label}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


# ─── Cached Model Loaders ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_cnn_model(backbone: str, model_path: str, device_str: str):
    """Load and cache the CNN model."""
    device = torch.device(device_str)
    backbone_lower = backbone.lower()

    if "vit-base" in backbone_lower or "vit_base" in backbone_lower:
        model = ViTAgePrediction("vit_base_patch16_224")
    elif "vit-tiny" in backbone_lower or "vit_tiny" in backbone_lower:
        model = ViTAgePrediction("vit_tiny_patch16_224")
    else:
        model = AgePredictionCNN(backbone=backbone_lower)

    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            logger.info(f"CNN loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load CNN weights from {model_path}: {e}. Using random weights.")
    else:
        logger.warning(f"CNN weights not found at {model_path}. Using random weights.")

    model.to(device).eval()
    return model, device


@st.cache_resource(show_spinner=False)
def load_yolo_model(yolo_path: str):
    """Load and cache the YOLO model."""
    if not HAS_YOLO:
        return None
    try:
        if os.path.exists(yolo_path):
            yolo = YOLO(yolo_path)
        else:
            logger.warning("Custom YOLO weights not found. Loading pretrained YOLOv8n.")
            yolo = YOLO("yolov8n.pt")
        return yolo
    except Exception as e:
        logger.warning(f"YOLO load failed: {e}")
        return None


# ─── Face Detection ────────────────────────────────────────────────────────
class MultiFaceDetector:
    def __init__(self, yolo_model=None):
        self.yolo = yolo_model
        self.cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            c = cv2.CascadeClassifier(cascade_path)
            if not c.empty():
                self.cascade = c
        except Exception:
            pass

    def detect(self, img_bgr: np.ndarray, conf: float = 0.35):
        """Returns list of {bbox, crop_bgr, confidence, method}."""
        faces = []

        if self.yolo is not None:
            try:
                res = self.yolo(img_bgr, verbose=False, conf=conf)
                h, w = img_bgr.shape[:2]
                for box in res[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    confidence = float(box.conf[0])
                    pad_x = int((x2 - x1) * 0.12)
                    pad_y = int((y2 - y1) * 0.12)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    if (x2 - x1) > 20 and (y2 - y1) > 20:
                        crop = img_bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            faces.append({"bbox": (x1, y1, x2, y2), "crop_bgr": crop,
                                          "confidence": confidence, "method": "YOLO"})
            except Exception as e:
                logger.warning(f"YOLO inference error: {e}")

        if not faces and self.cascade is not None:
            try:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                rects = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                for (x, y, fw, fh) in rects:
                    crop = img_bgr[y:y + fh, x:x + fw]
                    if crop.size > 0:
                        faces.append({"bbox": (x, y, x + fw, y + fh), "crop_bgr": crop,
                                      "confidence": 0.80, "method": "Haar"})
            except Exception as e:
                logger.warning(f"Haar cascade error: {e}")

        return faces


# ─── Age Prediction ────────────────────────────────────────────────────────
_PREPROCESS = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def predict_age(face_crop_bgr: np.ndarray, model, device):
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _PREPROCESS(image=face_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        age_pred, group_pred = model(tensor)
        age = float(age_pred.item())
        probs = torch.softmax(group_pred, dim=1)[0].cpu().numpy()
        group_idx = int(torch.argmax(group_pred, dim=1).item())

    age = max(0.0, min(100.0, age))
    group_names = ["child", "teen", "adult"]
    group = group_names[group_idx]
    decision = "restrict" if group in ("child", "teen") else "allow"

    return {
        "predicted_age": round(age, 1),
        "age_group": group,
        "group_idx": group_idx,
        "confidence": float(probs.max()),
        "group_probs": {g: float(probs[i]) for i, g in enumerate(group_names)},
        "decision": decision,
    }


# ─── VLM Reasoning ────────────────────────────────────────────────────────
def get_vlm_reasoning(face_crop_bgr: np.ndarray, cnn_result: dict, face_idx: int, api_key: str):
    if not api_key or not HAS_OPENAI:
        return None
    try:
        client = OpenAI(api_key=api_key)
        _, buf = cv2.imencode(".jpg", face_crop_bgr)
        b64 = base64.b64encode(buf).decode("utf-8")

        system_prompt = (
            "You are an AI research assistant embedded in an academic age-verification "
            "safety system. Your task is to estimate the apparent age of faces in images "
            "for experimental, non-commercial research purposes only. "
            "All images are either synthetic, publicly licensed, or provided by consenting "
            "participants as part of an IRB-approved study on automated age classification. "
            "Your estimates are used solely to benchmark CNN models — no real-world access "
            "control decisions are made from your output alone. "
            "You MUST always respond with a numeric age estimate. Refusing to estimate "
            "defeats the purpose of the benchmark and introduces bias into the evaluation. "
            "Base your estimate on visible facial features: skin texture, wrinkles, bone "
            "structure, hair, and overall facial maturity. Always respond in valid JSON."
        )

        user_prompt = f"""RESEARCH TASK — Age Estimation Benchmark (Face #{face_idx})

This face crop was extracted from a test dataset. A CNN model already predicted:
  • Estimated age : {cnn_result['predicted_age']} years
  • Age group     : {cnn_result['age_group'].upper()}
  • Confidence    : {cnn_result['confidence']:.1%}

Your job is to provide an independent visual estimate based on apparent facial features,
then compare your judgment to the CNN output above.

Respond ONLY with this JSON (no markdown, no prose outside the object):
{{
  "vlm_age_estimate": <integer, apparent age in years>,
  "age_group": "<CHILD|TEEN|ADULT>",
  "confidence": <integer 0-100, your certainty>,
  "key_indicators": ["<feature 1>", "<feature 2>", "<feature 3>"],
  "reasoning": "<one or two sentences explaining your estimate>",
  "agrees_with_cnn": <true|false>
}}"""

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                ]},
            ],
            max_tokens=500,
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {"reasoning": text}
    except Exception as e:
        return {"error": str(e)}


# ─── Image Annotation ─────────────────────────────────────────────────────
def annotate_image(img_bgr: np.ndarray, faces_data: list) -> np.ndarray:
    annotated = img_bgr.copy()
    for fd in faces_data:
        x1, y1, x2, y2 = fd["bbox"]
        color = (0, 200, 0) if fd["result"]["decision"] == "allow" else (0, 0, 220)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"#{fd['id']} {fd['result']['predicted_age']:.0f}yr ({fd['result']['age_group']})"
        lsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated, (x1, y1 - lsize[1] - 10), (x1 + lsize[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated


# ─── Group badge helper ───────────────────────────────────────────────────
_GROUP_EMOJI = {"child": "👶", "teen": "🧒", "adult": "🧑"}
_DECISION_COLOR = {"allow": "green", "restrict": "red"}


def face_card(face_data: dict, vlm, col):
    r = face_data["result"]
    emoji = _GROUP_EMOJI.get(r["age_group"], "🙂")
    decision_badge = "✅ ALLOW" if r["decision"] == "allow" else "🚫 RESTRICT"
    color = _DECISION_COLOR[r["decision"]]

    with col:
        crop_rgb = cv2.cvtColor(face_data["crop_bgr"], cv2.COLOR_BGR2RGB)
        st.image(crop_rgb, caption=f"Face #{face_data['id']}", use_container_width=True)
        st.markdown(f"**{emoji} Age:** {r['predicted_age']:.0f} yrs &nbsp; **Group:** {r['age_group'].title()}")
        st.markdown(f"**Decision:** :{color}[{decision_badge}]")
        st.markdown(f"**CNN Confidence:** {r['confidence']:.1%}")

        with st.expander("Group probabilities"):
            for g, p in r["group_probs"].items():
                st.progress(p, text=f"{g}: {p:.1%}")

        if vlm:
            with st.expander("🧠 VLM Reasoning", expanded=True):
                if "error" in vlm:
                    st.warning(f"VLM error: {vlm['error']}")
                elif "vlm_age_estimate" in vlm:
                    agree_icon = "✅" if vlm.get("agrees_with_cnn") else "⚠️"
                    agree_text = "Agrees with CNN" if vlm.get("agrees_with_cnn") else "Differs from CNN"
                    st.markdown(
                        f"**VLM Age:** {vlm['vlm_age_estimate']} yrs &nbsp;·&nbsp; "
                        f"**Group:** {vlm.get('age_group', '—')} &nbsp;·&nbsp; "
                        f"**Confidence:** {vlm.get('confidence', '—')}% &nbsp; {agree_icon} {agree_text}"
                    )
                    indicators = vlm.get("key_indicators", [])
                    if indicators:
                        st.markdown("**Key Indicators:**")
                        for ind in indicators:
                            st.markdown(f"&nbsp;&nbsp;• {ind}")
                    if vlm.get("reasoning"):
                        st.markdown(f"**Reasoning:** {vlm['reasoning']}")
                else:
                    st.json(vlm)


# ─── Live Webcam Video Processor (WebRTC) ────────────────────────────────
if HAS_WEBRTC:
    class AgeVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = None
            self.model    = None
            self.device   = None
            self.conf     = 0.35

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.detector is not None and self.model is not None:
                try:
                    faces = self.detector.detect(img, conf=self.conf)
                    for face in faces:
                        result = predict_age(face["crop_bgr"], self.model, self.device)
                        x1, y1, x2, y2 = face["bbox"]
                        color = (0, 200, 0) if result["decision"] == "allow" else (0, 0, 220)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        label = f"{result['predicted_age']:.0f}yr · {result['age_group']}"
                        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
                        cv2.rectangle(img, (x1, y1 - lsz[1] - 8), (x1 + lsz[0], y1), color, -1)
                        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                except Exception:
                    pass
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ─── Auto-download models on startup ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _ensure_models(model_dir: str):
    """Download all missing or corrupt model files from GitHub Releases at app startup."""
    os.makedirs(model_dir, exist_ok=True)
    needed = [
        (fname, url)
        for fname, url in Config.MODEL_URLS.items()
        if not _is_valid_model_file(os.path.join(model_dir, fname))
    ]
    if not needed:
        return
    st.info(f"📦 Downloading {len(needed)} model file(s) from GitHub Releases…")
    for fname, url in needed:
        dest = os.path.join(model_dir, fname)
        ok = _gh_download(url, dest, fname)
        if ok:
            logger.info(f"Downloaded {fname} → {dest}")
        else:
            logger.warning(f"Failed to download {fname}")


# ─── Main App ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Age Verification v2.0",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Age Verification System v2.0")
    st.caption("YOLO Multi-Face Detection → CNN Age Prediction → VLM Per-Face Reasoning")

    # Auto-download model weights from GitHub Releases if not present locally
    _ensure_models(Config.MODEL_DIR)

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Input Mode",
            ["📷 Image Upload", "📹 Video Upload", "🎥 Webcam"],
            index=0,
        )

        st.divider()
        use_vlm = st.toggle("🧠 VLM Reasoning (GPT-4V)", value=False)
        if use_vlm:
            api_key_input = st.text_input(
                "OpenAI API Key",
                value=Config.get_openai_key(),
                type="password",
                help="Required for VLM reasoning",
            )
        else:
            api_key_input = ""

        st.divider()
        backbone = st.selectbox(
            "CNN Backbone",
            ["densenet", "resnet50", "efficientnet_v2", "convnext", "vit-base", "vit-tiny"],
            index=0,
            help="Must match the backbone used during training",
        )
        conf_threshold = st.slider("Detection Confidence", 0.10, 0.90, 0.35, 0.05)

        st.divider()
        st.markdown("**Model Paths**")
        _default_cnn_file = Config.BACKBONE_FILES.get(backbone, Config.CNN_FILENAME)
        cnn_path_input = st.text_input(
            "CNN weights path",
            value=os.path.join(Config.MODEL_DIR, _default_cnn_file),
        )
        yolo_path_input = st.text_input(
            "YOLO weights path",
            value=os.path.join(Config.MODEL_DIR, Config.YOLO_FILENAME),
        )

        if st.button("⬇️  Re-download from GitHub Releases"):
            _cnn_url = Config.MODEL_URLS.get(os.path.basename(cnn_path_input), "")
            _yolo_url = Config.MODEL_URLS.get(os.path.basename(yolo_path_input), "")
            if _cnn_url:
                ok = _gh_download(_cnn_url, cnn_path_input, "CNN model")
                st.success("CNN downloaded ✅") if ok else st.error("CNN download failed ❌")
            if _yolo_url:
                ok = _gh_download(_yolo_url, yolo_path_input, "YOLO model")
                st.success("YOLO downloaded ✅") if ok else st.error("YOLO download failed ❌")

    # ── Load Models ───────────────────────────────────────────────────────
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    with st.spinner("Loading models …"):
        cnn_model, device = load_cnn_model(backbone, cnn_path_input, device_str)
        yolo_model = load_yolo_model(yolo_path_input)

    cnn_ok   = os.path.exists(cnn_path_input)
    yolo_ok  = yolo_model is not None

    st.info(
        f"Device: **{device_str.upper()}** &nbsp;|&nbsp; "
        f"CNN: {'✅' if cnn_ok else '⚠️ random weights'} &nbsp;|&nbsp; "
        f"YOLO: {'✅' if yolo_ok else '❌'} &nbsp;|&nbsp; "
        f"VLM: {'✅' if (use_vlm and api_key_input) else '—'}"
    )

    detector = MultiFaceDetector(yolo_model=yolo_model)

    # ── Modes ─────────────────────────────────────────────────────────────
    if mode == "📷 Image Upload":
        _image_mode(detector, cnn_model, device, conf_threshold, use_vlm, api_key_input)

    elif mode == "📹 Video Upload":
        _video_mode(detector, cnn_model, device, conf_threshold)

    elif mode == "🎥 Webcam":
        _webcam_mode(detector, cnn_model, device, conf_threshold, use_vlm, api_key_input)


# ─── Image Mode ──────────────────────────────────────────────────────────
def _image_mode(detector, model, device, conf, use_vlm, api_key):
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"], key="img_upload"
    )
    if not uploaded:
        return

    file_bytes = np.frombuffer(uploaded.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not decode image.")
        return

    with st.spinner("Detecting faces …"):
        raw_faces = detector.detect(img_bgr, conf=conf)

    if not raw_faces:
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Input", use_container_width=True)
        st.warning("⚠️  No faces detected. Try lowering the confidence threshold.")
        return

    st.success(f"✅ Detected **{len(raw_faces)}** face(s)")

    # Predict age for each face
    faces_data = []
    for i, face in enumerate(raw_faces):
        result = predict_age(face["crop_bgr"], model, device)
        vlm = get_vlm_reasoning(face["crop_bgr"], result, i + 1, api_key) if use_vlm else None
        faces_data.append({
            "id": i + 1,
            "bbox": face["bbox"],
            "crop_bgr": face["crop_bgr"],
            "result": result,
            "vlm": vlm,
        })

    annotated = annotate_image(img_bgr, faces_data)

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)

    st.divider()
    st.subheader("Per-Face Results")
    n_cols = min(len(faces_data), 4)
    cols = st.columns(n_cols)
    for i, fd in enumerate(faces_data):
        face_card(fd, fd.get("vlm"), cols[i % n_cols])

    # Summary table
    st.divider()
    st.subheader("Summary")
    rows = []
    for fd in faces_data:
        r = fd["result"]
        row = {
            "Face": f"#{fd['id']}",
            "Age (CNN)": f"{r['predicted_age']:.0f}",
            "Group": r["age_group"].title(),
            "Confidence": f"{r['confidence']:.1%}",
            "Decision": "✅ Allow" if r["decision"] == "allow" else "🚫 Restrict",
            "Det. Method": [f for f in [raw_faces[fd["id"]-1].get("method")] if f][0],
        }
        if fd.get("vlm") and "vlm_age_estimate" in fd["vlm"]:
            row["Age (VLM)"] = str(fd["vlm"]["vlm_age_estimate"])
        rows.append(row)
    st.dataframe(rows, use_container_width=True)


# ─── Video Mode ──────────────────────────────────────────────────────────
def _video_mode(detector, model, device, conf):
    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"], key="vid_upload"
    )
    if not uploaded:
        return

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name

    st.video(uploaded)

    max_frames = st.number_input("Max frames to process", 30, 500, 150, 10)

    if not st.button("🔬 Analyse Video"):
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Cannot open video.")
        return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    st.info(f"Video: {width}×{height} @ {fps:.0f} FPS — processing up to {total} frames")

    # Output writer — try codecs in order of headless-Linux compatibility
    out_path = video_path + "_annotated.mp4"
    writer = None
    for fourcc_str in ("avc1", "mp4v", "XVID", "MJPG"):
        try:
            w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height))
            if w.isOpened():
                writer = w
                break
            w.release()
        except Exception:
            pass
    if writer is None:
        st.error("Could not initialise video writer. OpenCV codec support may be missing.")
        cap.release()
        return

    progress_bar = st.progress(0)
    status_text  = st.empty()
    frame_count  = 0
    all_ages     = []
    sample_frames = []

    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 2 == 0:
                raw_faces = detector.detect(frame, conf=conf)
                faces_data = []
                for i, face in enumerate(raw_faces):
                    result = predict_age(face["crop_bgr"], model, device)
                    faces_data.append({"id": i+1, "bbox": face["bbox"], "result": result})
                    all_ages.append(result["predicted_age"])

                annotated = annotate_image(frame, faces_data)
                fps_text = f"Faces: {len(raw_faces)}"
                cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                writer.write(annotated)
                if frame_count % max(1, max_frames // 6) == 0:
                    sample_frames.append(annotated.copy())
            else:
                writer.write(frame)

            frame_count += 1
            progress_bar.progress(min(frame_count / max_frames, 1.0))
            status_text.text(f"Frame {frame_count}/{max_frames}")
    finally:
        cap.release()
        writer.release()

    st.success(f"✅ Processed {frame_count} frames | Total face detections: {len(all_ages)}")
    if all_ages:
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Predicted Age", f"{np.mean(all_ages):.1f} yrs")
        c2.metric("Min Age", f"{min(all_ages):.0f} yrs")
        c3.metric("Max Age", f"{max(all_ages):.0f} yrs")

    if sample_frames:
        n = min(len(sample_frames), 6)
        cols = st.columns(n)
        for i, fr in enumerate(sample_frames[:n]):
            cols[i].image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), caption=f"Sample {i+1}", use_container_width=True)

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            st.download_button("⬇️  Download Annotated Video", f, file_name="annotated_video.mp4", mime="video/mp4")


# ─── Webcam Mode ─────────────────────────────────────────────────────────
def _webcam_mode(detector, model, device, conf, use_vlm, api_key):
    tab_capture, tab_live = st.tabs(["📸 Capture & Analyse", "🎥 Live Stream"])

    # ── Tab 1: Capture ────────────────────────────────────────────────────
    with tab_capture:
        cam_image = st.camera_input("Take a photo then click Analyse")
        if not cam_image:
            st.info("Grant camera access and take a photo to analyse ages.")
        else:
            file_bytes = np.frombuffer(cam_image.getvalue(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not decode camera image.")
            else:
                with st.spinner("Analysing …"):
                    raw_faces = detector.detect(img_bgr, conf=conf)
                    faces_data = []
                    for i, face in enumerate(raw_faces):
                        result = predict_age(face["crop_bgr"], model, device)
                        vlm = get_vlm_reasoning(face["crop_bgr"], result, i+1, api_key) if use_vlm else None
                        faces_data.append({
                            "id": i+1, "bbox": face["bbox"],
                            "crop_bgr": face["crop_bgr"], "result": result, "vlm": vlm,
                        })

                if not faces_data:
                    st.warning("No faces detected. Try lowering the confidence threshold.")
                else:
                    annotated = annotate_image(img_bgr, faces_data)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                    with col2:
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detected", use_container_width=True)
                    st.divider()
                    st.subheader("Per-Face Results")
                    n_cols = min(len(faces_data), 4)
                    cols = st.columns(n_cols)
                    for i, fd in enumerate(faces_data):
                        face_card(fd, fd.get("vlm"), cols[i % n_cols])

    # ── Tab 2: Live Stream ────────────────────────────────────────────────
    with tab_live:
        if not HAS_WEBRTC:
            st.error("streamlit-webrtc is not installed. Run: pip install streamlit-webrtc aiortc av")
        else:
            st.info("Age detection runs on every frame in real-time. Green box = adult (allow), Red box = minor (restrict).")
            ctx = webrtc_streamer(
                key="age-live",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=AgeVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if ctx.video_processor:
                ctx.video_processor.detector = detector
                ctx.video_processor.model    = model
                ctx.video_processor.device   = device
                ctx.video_processor.conf     = conf


if __name__ == "__main__":
    main()
