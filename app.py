from flask import Flask, render_template, request, jsonify, send_file, abort, redirect, url_for, session
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'mpl-cache'))
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import joblib
import logging
from uuid import uuid4
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import threading
import urllib.request
import json
import secrets
from functools import lru_cache
from flask_sqlalchemy import SQLAlchemy
import bcrypt
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    PDF_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("ReportLab imported successfully for PDF generation")
except ImportError:
    PDF_AVAILABLE = False
    logger = logging.getLogger(__name__)

load_dotenv()
# Logging setup must come before LIME import to define logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# LIME imports (robust)
LIME_AVAILABLE = False
try:
    import lime
    from lime import lime_image
    from lime.lime_tabular import LimeTabularExplainer
    from skimage.segmentation import mark_boundaries, quickshift, slic
    LIME_AVAILABLE = True
    logger.debug("LIME imported successfully")
except Exception as e:
    logging.warning(f"LIME import failed: {e}. Please install with `pip install lime` if needed.")

# Gemini imports
GEMINI_AVAILABLE = False
try:
    from google import genai
    GEMINI_AVAILABLE = True
    logger.debug("Gemini imported successfully")
except Exception as e:
    logging.warning(f"Gemini import failed: {e}. Please install with `pip install google-genai` if needed.")

# App config
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# SQLite Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lungvision.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(60), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # Patient, Clinician, Researcher
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class LoginAudit(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recorded_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    event = db.Column(db.String(24), nullable=False)
    outcome = db.Column(db.String(16), nullable=False)
    actor_id = db.Column(db.Integer, nullable=True)

def record_login_audit(event, outcome, actor_id=None):
    """Persist a minimal audit record; never include credentials or account identifiers."""
    try:
        entry = LoginAudit(event=event, outcome=outcome, actor_id=actor_id)
        db.session.add(entry)
        db.session.commit()
        webhook = os.getenv('GOOGLE_SHEETS_WEBHOOK_URL')
        token = os.getenv('GOOGLE_SHEETS_WEBHOOK_TOKEN')
        if webhook and token:
            payload = json.dumps({
                'token': token,
                'recorded_at': entry.recorded_at.isoformat() + 'Z',
                'event': event,
                'outcome': outcome,
                'actor_id': actor_id,
            }).encode('utf-8')
            def deliver():
                try:
                    req = urllib.request.Request(webhook, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
                    urllib.request.urlopen(req, timeout=5).read(256)
                except Exception as exc:
                    logger.warning('Google Sheets audit delivery failed: %s', exc)
            threading.Thread(target=deliver, daemon=True).start()
    except Exception:
        db.session.rollback()
        logger.exception('Could not persist login audit record')

# Create DB tables if they don't exist
with app.app_context():
    db.create_all()

# Performance-related defaults
DEFAULT_LIME_NUM_SAMPLES = 100  
LIME_NUM_FEATURES = 8  # Increased for better feature coverage
MAX_PROCESS_TIME = 30

# Load scaler
SCALER_PATH = "scaler.pkl"
if not os.path.exists(SCALER_PATH):
    logger.error(f"Scaler file not found: {SCALER_PATH}. Place your scaler.pkl in the working folder.")
else:
    scaler = joblib.load(SCALER_PATH)
    logger.debug("Scaler loaded successfully. The scaler was serialized with scikit-learn 1.2.2; compatibility warnings may occur with newer versions.")

logger.debug(f"Current working directory: {os.getcwd()}")

# Model transforms
model_transforms = {
    'EfficientNetB3': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'DenseNet121': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'InceptionV3': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'ResNet50': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Model filenames
model_paths = {
    'ct': {
        'EfficientNetB3': ('models/ct_models/best_lung_ct_efficientnet_b3.pth', models.efficientnet_b3, 'features'),
        'DenseNet121': ('models/ct_models/best_lung_ct_densenet121.pth', models.densenet121, 'features.denseblock4'),
        'InceptionV3': ('models/ct_models/best_lung_ct_inception_v3.pth', models.inception_v3, 'Mixed_7c'),
        'ResNet50': ('models/ct_models/best_lung_ct_resnet50.pth', models.resnet50, 'layer4'),
    },
    'xray': {
        'EfficientNetB3': ('models/xray models/best_lung_efficientnet_b3.pth', models.efficientnet_b3, 'features'),
        'DenseNet121': ('models/xray models/best_lung_densenet121.pth', models.densenet121, 'features.denseblock4'),
        'InceptionV3': ('models/xray models/best_lung_inception_v3.pth', models.inception_v3, 'Mixed_7c'),
        'ResNet50': ('models/xray models/best_lung_resnet50.pth', models.resnet50, 'layer4'),
    }
}

metadata_model_path = 'models/csv_best_model.pth'

class_names = {
    # Must match the order of TISSUE_CLASSES + UNKNOWN_CLASS in the retrained CT notebooks.
    'ct': ["Normal", "Benign", "Malignant", "UNKNOWN"],
    'xray': [
        "00 Anatomia Normal",
        "01 Processos Inflamatórios Pulmonares",
        "02 Maior Densidade",
        "03 Menor Densidade",
        "04 Doenças Pulmonares Obstrutivas",
        "05 Doenças Infecciosas Degenerativas",
        "06 Lesões Encapsuladas",
        "07 Alterações de Mediastino",
        "08 Alterações do Tórax",
        "UNKNOWN",
    ],
    'metadata': ["YES", "NO"]
}

metadata_fields = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Preload models
preloaded_models = {}
def preload_models():
    for scan_type in model_paths:
        preloaded_models[scan_type] = {}
        for name, (path, arch_fn, _) in model_paths[scan_type].items():
            try:
                model = load_image_model(path, arch_fn, name, num_classes=len(class_names[scan_type]))
                model = model.to(device)
                preloaded_models[scan_type][name] = model
                logger.debug(f"Preloaded model {name} for {scan_type}")
            except Exception as e:
                logger.error(f"Failed to preload model {name} for {scan_type}: {e}")

def load_image_model(model_path, model_arch, model_name, num_classes=3):
    logger.debug(f"Loading image model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = model_arch(weights=None)
    if isinstance(model, models.DenseNet):
        in_features = model.classifier.in_features
        model.classifier = (nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
                            if num_classes == 10 else nn.Linear(in_features, num_classes))
    elif isinstance(model, models.EfficientNet):
        try:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        except Exception:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif isinstance(model, models.Inception3):
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif isinstance(model, models.ResNet):
        in_features = model.fc.in_features
        model.fc = (nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
                    if num_classes == 10 else nn.Linear(in_features, num_classes))
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_metadata_model(model_path, input_size=15):
    logger.debug(f"Loading metadata model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Metadata model not found: {model_path}")
    class MetadataModel(nn.Module):
        def __init__(self, input_size):
            super(MetadataModel, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2)
            )
        def forward(self, x):
            return self.net(x)
    model = MetadataModel(input_size=input_size)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    logger.debug(f"Preprocessing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((512, 512), Image.Resampling.LANCZOS)  # Early resizing
    tensor = transform(image).unsqueeze(0)
    return tensor, image

def preprocess_metadata(form_data):
    logger.debug(f"Preprocessing metadata: {form_data}")
    features = []
    for field in metadata_fields:
        value = form_data.get(field, 'No')
        if field == "GENDER":
            # Matches the LabelEncoder used during training: Female=0, Male=1.
            features.append(1 if value in ["M", "m", "Male", "male"] else 0)
        elif field == "AGE":
            try:
                features.append(float(value))
            except Exception:
                features.append(0.0)
        else:
            features.append(2 if str(value).strip().lower() in ['yes', '2', 'true', 'y'] else 1)
    features = np.array(features).reshape(1, -1)
    if 'scaler' in globals():
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    return torch.tensor(features_scaled.flatten(), dtype=torch.float32).unsqueeze(0)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        try:
            self.hooks.append(self.target_layer.register_forward_hook(self._save_activation))
            self.hooks.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        except Exception as e:
            logger.warning(f"GradCAM hook registration warning: {e}")

    def _save_activation(self, module, input, output):
        try:
            self.activations = output.detach().clone()
        except Exception:
            self.activations = None

    def _save_gradient(self, module, grad_input, grad_output):
        try:
            self.gradients = grad_output[0].detach().clone()
        except Exception:
            self.gradients = None

    def generate(self, input_tensor, class_idx):
        try:
            self.model.zero_grad()
            x = input_tensor.clone().detach().requires_grad_(True).to(device)
            output = self.model(x)
            if isinstance(output, tuple):
                output = output[0]
            loss = output[0, class_idx]
            loss.backward()
            if self.gradients is None or self.activations is None:
                raise RuntimeError("GradCAM: missing gradients or activations")
            pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
            activation = self.activations.squeeze(0).clone()
            for i in range(pooled_gradients.shape[0]):
                activation[i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activation, dim=0).cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (np.max(heatmap) + 1e-8)
            return heatmap
        except Exception as e:
            logger.error(f"Failed to generate Grad-CAM: {e}")
            return None

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []

    def __del__(self):
        self.remove_hooks()

def make_predict_fn_image_optimized(model, transform, device_str):
    """Optimized predict function for LIME with faster processing"""
    model.eval()
    
    def predict_fn(images):
        try:
            if len(images) == 0:
                return np.zeros((0, 3))
            
            # Process images in batches for efficiency
            batch_size = min(16, len(images))
            all_probs = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                tensors = []
                
                for img in batch_images:
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img)
                    else:
                        pil_img = img
                    
                    tensor = transform(pil_img)
                    tensors.append(tensor)
                
                if tensors:
                    batch_tensor = torch.stack(tensors).to(device)
                    with torch.no_grad():
                        outputs = model(batch_tensor)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        all_probs.append(batch_probs)
            
            if all_probs:
                return np.vstack(all_probs)
            else:
                return np.ones((len(images), 3)) / 3
                
        except Exception as e:
            logger.error(f"LIME predict_fn error: {e}")
            return np.ones((len(images), 3)) / 3
    
    return predict_fn

def make_predict_fn_tabular(model, device=torch.device('cpu')):
    model = model.to(device)
    model.eval()
    def predict_fn(X):
        try:
            Xt = torch.tensor(np.array(X), dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(Xt)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs
        except Exception as e:
            logger.error(f"LIME tabular predict_fn error: {e}")
            n = len(X)
            C = 2
            fallback = np.ones((n, C)) / C
            return fallback
    return predict_fn

def overlay_heatmap(img_pil, heatmap):
    try:
        img = np.array(img_pil)
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_normalized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
        return overlay
    except Exception as e:
        logger.error(f"Failed to overlay heatmap: {e}")
        return None
    
def generate_clinical_insights(consensus_label, consensus_conf, scan_type, metadata_result=None):
    """
    Generate professional clinical insights and recommendations based on diagnosis results.
    
    Args:
        consensus_label: The predicted label from consensus analysis
        consensus_conf: Confidence score of the prediction
        scan_type: Type of scan ('xray', 'ct', or 'metadata')
        metadata_result: Optional metadata analysis result with LIME explanations
    
    Returns:
        Dictionary containing insights and recommendations
    """
    insights = {
        'title': 'Clinical Insights & Recommendations',
        'analysis': '',
        'key_findings': [],
        'recommendations': [],
        'disclaimer': 'This system is designed for research and educational purposes only. Do not solely depend on these results. Please consult with qualified medical professionals or oncologists for proper diagnosis and treatment.',
        'severity': 'normal'  # 'normal', 'concern', or 'urgent'
    }
    
    confidence_pct = consensus_conf * 100
    if scan_type == 'xray':
        insights['title'] = 'Chest X-ray Model Output'
        insights['severity'] = 'informational'
        insights['analysis'] = (
            f"The X-ray ensemble's highest-scoring training category is {consensus_label} "
            f"({confidence_pct:.1f}% model confidence). This checkpoint was trained to classify "
            "nine Portuguese lung/thorax image categories plus UNKNOWN; it is not a lung-cancer "
            "classifier and this output is not a clinical diagnosis."
        )
        insights['key_findings'] = [
            f"Top model category: {consensus_label} ({confidence_pct:.1f}%).",
            "The displayed probabilities are averaged across the X-ray models that loaded successfully.",
        ]
        insights['recommendations'] = [
            "Have the original image and this research output reviewed by a qualified clinician.",
            "Do not use this model output to rule cancer in or out.",
        ]
        return insights

    if scan_type == 'ct':
        insights['title'] = 'CT Research Model Output'
        insights['severity'] = 'informational'
        insights['analysis'] = (
            f"The ensemble's highest-scoring training category is {consensus_label} "
            f"({confidence_pct:.1f}% model confidence). The CT models classify images into "
            "Normal, Benign, Malignant, and UNKNOWN training categories. This research output "
            "cannot confirm or rule out cancer and is not a diagnosis."
        )
        insights['key_findings'] = [
            f"Highest-scoring model category: {consensus_label} ({confidence_pct:.1f}%).",
            "The displayed probabilities are averaged across CT models that loaded successfully.",
        ]
        insights['recommendations'] = [
            "Discuss the original CT images and radiology report with a qualified healthcare professional.",
            "Do not use this model output to diagnose disease or make treatment decisions.",
        ]
        return insights

    # Normalize label for comparison
    is_cancer = any(term in consensus_label.lower() for term in ['cancer', 'yes'])
    
    if scan_type in ['xray', 'ct']:
        # Insights for CT/X-ray (ensemble of four models)
        if is_cancer:
            insights['severity'] = 'urgent' if confidence_pct > 70 else 'concern'
            insights['analysis'] = f"""Based on the comprehensive analysis of your {scan_type.upper()} scan, 
            our ensemble AI models have detected patterns consistent with lung cancer with {confidence_pct:.1f}% confidence. 
            Multiple deep learning models (EfficientNetB3, DenseNet121, InceptionV3, and ResNet50) have reached consensus on this finding."""
            
            insights['key_findings'] = [
                f"Consensus prediction indicates lung cancer presence ({confidence_pct:.1f}% confidence)",
                f"Multiple AI models show agreement in their independent analyses",
                "Visual explainability maps (GradCAM & LIME) highlight regions of concern in the imaging"
            ]
            
            insights['recommendations'] = [
                "🏥 <strong>Immediate Action:</strong> Schedule an appointment with an oncologist or pulmonologist as soon as possible",
                "📋 <strong>Further Testing:</strong> Request additional diagnostic procedures such as biopsy, PET scan, or molecular testing",
                "👨‍⚕️ <strong>Second Opinion:</strong> Consider consulting multiple specialists for comprehensive evaluation",
                "📊 <strong>Documentation:</strong> Bring these AI analysis results along with original imaging to your medical appointments",
                "💪 <strong>Stay Positive:</strong> Early detection significantly improves treatment outcomes. Modern therapies offer many effective options",
                "👨‍👩‍👧‍👦 <strong>Support System:</strong> Reach out to family, friends, and support groups during this time"
            ]
        else:
            insights['severity'] = 'normal'
            insights['analysis'] = f"""Our comprehensive AI analysis of your {scan_type.upper()} scan 
            shows <strong>no significant indicators of lung cancer</strong> at this time ({confidence_pct:.1f}% confidence for normal findings). 
            The ensemble of deep learning models reached consensus that the imaging patterns fall within normal parameters."""
            
            insights['key_findings'] = [
                f"Consensus prediction indicates normal/healthy lung tissue ({confidence_pct:.1f}% confidence)",
                "No suspicious patterns detected across multiple AI model analyses",
                "Visual explainability confirms absence of concerning regions"
            ]
            
            insights['recommendations'] = [
                "✅ <strong>Encouraging Results:</strong> These findings are reassuring, but continue regular health monitoring",
                "🔄 <strong>Routine Screening:</strong> Maintain annual health check-ups and follow recommended screening guidelines",
                "🚭 <strong>Prevention:</strong> Avoid smoking and secondhand smoke exposure to maintain lung health",
                "🏃 <strong>Healthy Lifestyle:</strong> Regular exercise, balanced diet, and adequate sleep support overall wellness",
                "⚠️ <strong>Watch for Symptoms:</strong> Consult a doctor if you develop persistent cough, chest pain, or breathing difficulties",
                "📅 <strong>Follow-up:</strong> Discuss these results with your primary care physician during your next visit"
            ]
    
    else:  # metadata
        # Insights for metadata (single MLP model)
        if is_cancer:
            insights['severity'] = 'urgent' if confidence_pct > 70 else 'concern'
            insights['analysis'] = f"""Based on the analysis of your clinical data, 
            our AI model has detected a high likelihood of lung cancer with {confidence_pct:.1f}% confidence. 
            The model identified significant risk factors contributing to this prediction."""
            
            insights['key_findings'] = [
                f"Prediction indicates lung cancer risk ({confidence_pct:.1f}% confidence)",
                "Clinical symptom analysis highlights elevated risk based on input data"
            ]
            
            # Add specific symptoms contributing to the prediction
            risk_factors = []
            meta_input = metadata_result.get('input', {}) if metadata_result else {}
            
            if meta_input.get('SMOKING') == 'Yes':
                risk_factors.append('smoking history')
            if meta_input.get('YELLOW_FINGERS') == 'Yes':
                risk_factors.append('nicotine staining')
            if meta_input.get('CHRONIC DISEASE') == 'Yes':
                risk_factors.append('pre-existing chronic conditions')
            if meta_input.get('SHORTNESS OF BREATH') == 'Yes':
                risk_factors.append('respiratory symptoms')
            if meta_input.get('CHEST PAIN') == 'Yes':
                risk_factors.append('chest discomfort')
            if meta_input.get('COUGHING') == 'Yes':
                risk_factors.append('persistent cough')
                
            if risk_factors:
                insights['key_findings'].append(f"Major symptoms contributing to risk: {', '.join(risk_factors)}")
            
            insights['recommendations'] = [
                "🏥 <strong>Immediate Action:</strong> Consult an oncologist or pulmonologist urgently",
                "📋 <strong>Further Testing:</strong> Request imaging (CT/X-ray), biopsy, or other diagnostics",
                "👨‍⚕️ <strong>Medical Evaluation:</strong> Share these results with your doctor for a thorough assessment",
                "🚭 <strong>Stop Smoking:</strong> Cease smoking immediately to reduce further risk",
                "📊 <strong>Monitor Symptoms:</strong> Track any worsening of symptoms like cough or chest pain",
                "👨‍👩‍👧‍👦 <strong>Support System:</strong> Engage family or support groups for emotional support"
            ]
        else:
            insights['severity'] = 'normal'
            insights['analysis'] = f"""Based on the analysis of your clinical data, 
            our AI model indicates <strong>no significant risk of lung cancer</strong> at this time ({confidence_pct:.1f}% confidence for low risk). 
            The model found minimal risk factors in the provided symptoms."""
            
            insights['key_findings'] = [
                f"Prediction indicates low lung cancer risk ({confidence_pct:.1f}% confidence)",
                "Clinical symptom analysis shows no major indicators of malignancy"
            ]
            
            insights['recommendations'] = [
                "✅ <strong>Encouraging Results:</strong> Your symptom profile suggests low risk, but stay vigilant",
                "🔄 <strong>Routine Check-ups:</strong> Continue regular medical screenings as recommended",
                "🚭 <strong>Prevention:</strong> Avoid smoking and exposure to secondhand smoke",
                "🏃 <strong>Healthy Lifestyle:</strong> Maintain a balanced diet, exercise, and healthy habits",
                "⚠️ <strong>Monitor Symptoms:</strong> Report new symptoms like persistent cough or chest pain to your doctor",
                "📅 <strong>Follow-up:</strong> Discuss these results with your physician at your next visit"
            ]
    
    return insights



def create_enhanced_lime_overlay(image_pil, mask, save_path):
    """Enhanced LIME overlay with better visualization"""
    try:
        image_array = np.array(image_pil)
        
        # Create a more sophisticated overlay
        overlay = image_array.copy().astype(np.float32)
        
        # Normalize mask to better range
        mask_normalized = mask.copy()
        if np.max(np.abs(mask_normalized)) > 0:
            mask_normalized = mask_normalized / np.max(np.abs(mask_normalized))
        
        # Create positive and negative regions
        positive_mask = mask_normalized > 0
        negative_mask = mask_normalized < 0
        
        # Apply strong highlighting for important regions
        if np.any(positive_mask):
            # Green for positive regions (supporting prediction)
            overlay[positive_mask] = overlay[positive_mask] * 0.3 + np.array([0, 255, 0]) * 0.7
        
        if np.any(negative_mask):
            # Red for negative regions (opposing prediction)  
            overlay[negative_mask] = overlay[negative_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
        
        # Convert back to uint8
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Save the image
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return save_path
        
    except Exception as e:
        logger.error(f"Failed to create enhanced LIME overlay: {e}")
        return None

def get_optimal_segmentation(image_array, scan_type):
    """Get optimal segmentation parameters based on scan type"""
    try:
        if scan_type == 'xray':
            # For X-rays, use SLIC with parameters optimized for chest X-rays
            segments = slic(image_array, n_segments=150, compactness=10, sigma=1, start_label=1)
        else:  # CT scans
            # For CT scans, use quickshift which works better with medical CT images
            segments = quickshift(image_array, kernel_size=3, max_dist=6, ratio=0.5)
        
        return segments
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        # Fallback to simple SLIC
        try:
            return slic(image_array, n_segments=100, compactness=10, sigma=1, start_label=1)
        except:
            # Ultimate fallback - create simple grid segments
            h, w = image_array.shape[:2]
            segments = np.zeros((h, w), dtype=np.int32)
            seg_h, seg_w = h // 10, w // 10
            for i in range(10):
                for j in range(10):
                    segments[i*seg_h:(i+1)*seg_h, j*seg_w:(j+1)*seg_w] = i*10 + j
            return segments

def enhanced_prediction_with_explanations(image_path, model, model_name, transform, target_layer, class_names_list,
                                         scan_type, lime_num_samples=DEFAULT_LIME_NUM_SAMPLES):
    results = {}
    start_time = time.time()
    logger.info(f"Starting prediction for {model_name} on {image_path}")
    try:
        # Preprocess
        preprocess_start = time.time()
        input_tensor, original_img = preprocess_image(image_path, transform)
        logger.info(f"Preprocessing took {time.time() - preprocess_start:.2f}s")

        # Model inference
        inference_start = time.time()
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.to(device))
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1).squeeze(0)
        conf, pred = torch.max(probs, 0)
        class_probs = {class_names_list[i]: probs[i].item() for i in range(len(class_names_list))}
        logger.info(f"Inference took {time.time() - inference_start:.2f}s")

        # GradCAM
        gradcam_start = time.time()
        gradcam = GradCAM(model, target_layer)
        gradcam_filename = None
        try:
            heatmap = gradcam.generate(input_tensor.to(device), pred.item())
            if heatmap is not None:
                overlay = overlay_heatmap(original_img, heatmap)
                if overlay is not None:
                    gradcam_filename = f"{uuid4()}_gradcam.jpg"
                    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                    if overlay.dtype != np.uint8:
                        overlay = ((overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8) * 255).astype(np.uint8)
                    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        except Exception as e:
            logger.error(f"GradCAM failure for {model_name}: {e}")
        finally:
            gradcam.remove_hooks()
        logger.info(f"GradCAM took {time.time() - gradcam_start:.2f}s")

        results.update({
            'name': model_name,
            'label': class_names_list[pred.item()],
            'confidence': float(conf.item()),
            'class_probs': class_probs,
            'gradcam_filename': gradcam_filename,
            'overlay_filename': gradcam_filename
        })

        # Enhanced LIME
        lime_start = time.time()
        lime_filename = None
        if LIME_AVAILABLE:
            try:
                # Optimize samples based on device and time constraints
                effective_samples = lime_num_samples if device.type == 'cuda' else max(50, lime_num_samples // 2)
                
                # Create optimized predict function
                predict_fn = make_predict_fn_image_optimized(model, transform, str(device))
                
                # Initialize LIME explainer
                explainer = lime_image.LimeImageExplainer()
                
                # Prepare image
                img_np = np.array(original_img)
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
                
                # Get optimal segmentation
                def custom_segmentation(image):
                    return get_optimal_segmentation(image, scan_type)
                
                # Generate explanation
                explanation = explainer.explain_instance(
                    img_np,
                    predict_fn,
                    top_labels=1,
                    hide_color=0,
                    num_samples=effective_samples,
                    num_features=LIME_NUM_FEATURES,
                    segmentation_fn=custom_segmentation
                )
                
                if explanation is not None:
                    top_label = explanation.top_labels[0]
                    temp, mask = explanation.get_image_and_mask(
                        top_label,
                        positive_only=False,
                        num_features=LIME_NUM_FEATURES,
                        hide_rest=False,
                        min_weight=0.01  # Lower threshold to capture more features
                    )
                    
                    lime_filename = f"{uuid4()}_lime.jpg"
                    lime_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_filename)
                    saved = create_enhanced_lime_overlay(original_img, mask, lime_path)
                    if not saved:
                        lime_filename = None
                else:
                    logger.error(f"LIME explanation is None for {model_name}")
                    
            except Exception as e:
                logger.error(f"LIME image explanation failed for {model_name}: {e}")
                lime_filename = None
                
        results['lime_filename'] = lime_filename
        logger.info(f"LIME took {time.time() - lime_start:.2f}s")

        processing_time = time.time() - start_time
        logger.info(f"Total processing time for {model_name}: {processing_time:.2f}s")
        if processing_time > MAX_PROCESS_TIME:
            logger.warning(f"Processing time ({processing_time:.2f}s) exceeded target ({MAX_PROCESS_TIME}s) for {model_name}")
        return results
        
    except Exception as e:
        logger.error(f"Enhanced prediction failed for {model_name}: {e}")
        return {
            'name': model_name,
            'error': str(e),
            'gradcam_filename': None,
            'lime_filename': None,
            'overlay_filename': None
        }

def generate_pdf_report(report_data, report_type='image'):
    """
    Generate a comprehensive PDF report for lung cancer diagnosis
    
    Args:
        report_data: Dictionary containing all report data
        report_type: 'image' for X-ray/CT scans, 'metadata' for clinical data
    
    Returns:
        Path to generated PDF file
    """
    if not PDF_AVAILABLE:
        logger.error("ReportLab not available for PDF generation")
        return None
    
    try:
        # Create unique filename
        report_filename = f"LungVision_Report_{uuid4()}.pdf"
        report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(report_path, pagesize=letter,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for PDF elements
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2C3E50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495E'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#5D6D7E'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY
        )
        
        # Add header
        story.append(Paragraph("🫁 LungVision AI", title_style))
        story.append(Paragraph("Multimodal Lung Cancer Detection Report", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        # Add report metadata
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {current_date}", normal_style))
        story.append(Paragraph(f"<b>Report ID:</b> {report_filename.replace('.pdf', '')}", normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        if report_type == 'image':
            # IMAGE REPORT (X-ray/CT scan)
            story.append(Paragraph("Imaging Analysis Report", heading_style))
            story.append(Paragraph(f"<b>Scan Type:</b> {report_data.get('scan_type', 'Unknown').upper()}", normal_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Add original image if available
            if report_data.get('original_image_path'):
                try:
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], report_data['original_image_path'])
                    if os.path.exists(img_path):
                        story.append(Paragraph("Original Medical Image", subheading_style))
                        img = RLImage(img_path, width=3*inch, height=3*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    logger.error(f"Error adding original image to PDF: {e}")
            
            # Individual Model Results
            story.append(Paragraph("Individual Model Predictions", heading_style))
            
            if report_data.get('model_results'):
                for idx, result in enumerate(report_data['model_results']):
                    story.append(Paragraph(f"<b>Model {idx + 1}: {result['name']}</b>", subheading_style))
                    
                    # Model predictions table
                    model_data = [
                        ['Prediction', result['label']],
                        ['Confidence', f"{result['confidence'] * 100:.1f}%"]
                    ]
                    
                    model_table = Table(model_data, colWidths=[2*inch, 3*inch])
                    model_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ECF0F1')),
                        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                        ('TOPPADDING', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    story.append(model_table)
                    story.append(Spacer(1, 0.1*inch))
                    
                    # Add class probabilities
                    if result.get('class_probs'):
                        story.append(Paragraph("<b>Class Probabilities:</b>", normal_style))
                        prob_data = [['Class', 'Probability']]
                        for class_name, prob in result['class_probs'].items():
                            display_name = "Lung Cancer" if "cancer" in class_name.lower() else class_name
                            prob_data.append([display_name, f"{prob * 100:.1f}%"])
                        
                        prob_table = Table(prob_data, colWidths=[2.5*inch, 2*inch])
                        prob_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('FONTSIZE', (0, 0), (-1, -1), 9),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                            ('TOPPADDING', (0, 0), (-1, -1), 6),
                            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                        ]))
                        story.append(prob_table)
                    
                    story.append(Spacer(1, 0.2*inch))
            
            # Consensus Analysis
            story.append(PageBreak())
            story.append(Paragraph("Ensemble Consensus Analysis", heading_style))
            
            consensus_data = [
                ['Consensus Prediction', report_data.get('consensus_label', 'N/A')],
                ['Consensus Confidence', f"{report_data.get('consensus_conf', 0) * 100:.1f}%"]
            ]
            
            consensus_table = Table(consensus_data, colWidths=[2.5*inch, 3*inch])
            consensus_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F8F5')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 2, colors.HexColor('#16A085'))
            ]))
            story.append(consensus_table)
            story.append(Spacer(1, 0.2*inch))
            
            # Consensus probabilities
            if report_data.get('consensus_class_probs'):
                story.append(Paragraph("<b>Consensus Class Probabilities:</b>", subheading_style))
                consensus_prob_data = [['Class', 'Probability']]
                for class_name, prob in report_data['consensus_class_probs'].items():
                    display_name = "Lung Cancer" if "cancer" in class_name.lower() else class_name
                    consensus_prob_data.append([display_name, f"{prob * 100:.1f}%"])
                
                consensus_prob_table = Table(consensus_prob_data, colWidths=[2.5*inch, 2*inch])
                consensus_prob_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16A085')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                story.append(consensus_prob_table)
                story.append(Spacer(1, 0.3*inch))
            
            story.append(Paragraph("<i>Note: The consensus prediction represents the average of all model predictions, providing a robust diagnosis.</i>", normal_style))
            story.append(Spacer(1, 0.3*inch))
            
        else:
            # METADATA REPORT (Clinical Data)
            story.append(Paragraph("Clinical Data Analysis Report", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Prediction result
            story.append(Paragraph("Prediction Summary", subheading_style))
            prediction_data = [
                ['Prediction', report_data.get('label', 'N/A')],
                ['Confidence', f"{report_data.get('confidence', 0) * 100:.1f}%"]
            ]
            
            prediction_table = Table(prediction_data, colWidths=[2.5*inch, 3*inch])
            prediction_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F8F5')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 2, colors.HexColor('#16A085'))
            ]))
            story.append(prediction_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Clinical data input
            story.append(Paragraph("Patient Clinical Data", subheading_style))
            
            if report_data.get('input'):
                clinical_data = [['Parameter', 'Value']]
                for field, value in report_data['input'].items():
                    clinical_data.append([field.replace('_', ' '), str(value)])
                
                clinical_table = Table(clinical_data, colWidths=[3*inch, 2*inch])
                clinical_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ECF0F1')])
                ]))
                story.append(clinical_table)
                story.append(Spacer(1, 0.3*inch))
            
            # Add LIME image if available
            if report_data.get('lime_filename'):
                try:
                    lime_path = os.path.join(app.config['UPLOAD_FOLDER'], report_data['lime_filename'])
                    if os.path.exists(lime_path):
                        story.append(PageBreak())
                        story.append(Paragraph("LIME Feature Importance Analysis", subheading_style))
                        story.append(Paragraph("This chart shows which clinical factors most influenced the prediction.", normal_style))
                        story.append(Spacer(1, 0.1*inch))
                        lime_img = RLImage(lime_path, width=5*inch, height=3.5*inch)
                        story.append(lime_img)
                        story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    logger.error(f"Error adding LIME image to PDF: {e}")
        
        # Clinical Insights Section (common for both)
        if report_data.get('clinical_insights'):
            story.append(PageBreak())
            insights = report_data['clinical_insights']
            
            story.append(Paragraph("Clinical Insights & Recommendations", heading_style))
            
            # Analysis
            if insights.get('analysis'):
                story.append(Paragraph("<b>Analysis:</b>", subheading_style))
                # Remove HTML tags for PDF
                analysis_text = insights['analysis'].replace('<strong>', '').replace('</strong>', '')
                story.append(Paragraph(analysis_text, normal_style))
                story.append(Spacer(1, 0.2*inch))
            
            # Key Findings
            if insights.get('key_findings'):
                story.append(Paragraph("<b>Key Findings:</b>", subheading_style))
                for finding in insights['key_findings']:
                    clean_finding = finding.replace('<strong>', '').replace('</strong>', '')
                    story.append(Paragraph(f"• {clean_finding}", normal_style))
                story.append(Spacer(1, 0.2*inch))
            
            # Recommendations
            if insights.get('recommendations'):
                story.append(Paragraph("<b>Recommendations:</b>", subheading_style))
                for recommendation in insights['recommendations']:
                    # Clean HTML tags
                    clean_rec = recommendation
                    clean_rec = re.sub(r'<[^>]+>', '', clean_rec)
                    story.append(Paragraph(f"• {clean_rec}", normal_style))
                story.append(Spacer(1, 0.2*inch))
            
            # Disclaimer
            if insights.get('disclaimer'):
                disclaimer_style = ParagraphStyle(
                    'Disclaimer',
                    parent=styles['Normal'],
                    fontSize=9,
                    textColor=colors.HexColor('#C0392B'),
                    leading=12,
                    alignment=TA_JUSTIFY,
                    borderWidth=1,
                    borderColor=colors.HexColor('#E74C3C'),
                    borderPadding=10,
                    backColor=colors.HexColor('#FADBD8')
                )
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph(f"<b>⚕️ IMPORTANT DISCLAIMER:</b><br/>{insights['disclaimer']}", disclaimer_style))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Paragraph("This report is generated by LungVision AI for research and educational purposes.", footer_style))
        story.append(Paragraph("© 2025 LungVision AI - All Rights Reserved", footer_style))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated successfully: {report_filename}")
        return report_filename
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None
    

@app.errorhandler(413)
def request_entity_too_large(error):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
    return render_template('diagnosis.html', error="File too large. Maximum size is 16MB.", metadata_fields=metadata_fields), 413

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    return "Image not found", 404

@app.route('/download_report', methods=['POST'])
def download_report():
    """Generate and download PDF report"""
    if not PDF_AVAILABLE:
        return jsonify({'error': 'PDF generation not available. Please install reportlab.'}), 500
    
    try:
        report_data = request.json
        report_type = report_data.get('report_type', 'image')
        
        # Generate PDF
        report_filename = generate_pdf_report(report_data, report_type)
        
        if report_filename:
            report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
            return send_file(
                report_path,
                as_attachment=True,
                download_name=f"LungVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': 'Failed to generate report'}), 500
            
    except Exception as e:
        logger.error(f"Error in download_report route: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_metadata_form', methods=['POST'])
def predict_metadata_form():
    logger.debug("Entered /predict_metadata_form route")
    try:
        form_data = {field: request.form.get(field) for field in metadata_fields}
        for field in metadata_fields:
            if form_data.get(field) is None:
                return render_template('diagnosis.html', error=f"Missing required field: {field}", metadata_fields=metadata_fields)

        input_tensor = preprocess_metadata(form_data)
        metadata_model = load_metadata_model(metadata_model_path)
        metadata_model = metadata_model.to(device)
        with torch.no_grad():
            output = metadata_model(input_tensor.to(device))
            probs = torch.softmax(output, dim=1)
            raw_output = probs[0][1].item()
            predicted_label = "YES" if raw_output >= 0.5 else "NO"
            confidence = raw_output if raw_output >= 0.5 else 1.0 - raw_output

        metadata_result = {
            'label': predicted_label,
            'confidence': confidence,
            'input': form_data,
            'raw_output': raw_output
        }

        # Generate insights for metadata
        clinical_insights_metadata = generate_clinical_insights(
            consensus_label=predicted_label,
            consensus_conf=confidence,
            scan_type='metadata',
            metadata_result=metadata_result
        )

        if LIME_AVAILABLE:
            try:
                lime_explainer = LimeTabularExplainer(
                    training_data=np.random.randn(100, len(metadata_fields)),
                    feature_names=metadata_fields,
                    class_names=['No Cancer', 'Cancer'],
                    mode='classification'
                )
                features_array = input_tensor.cpu().numpy()
                explanation = lime_explainer.explain_instance(features_array[0], make_predict_fn_tabular(metadata_model, device=device), num_features=len(metadata_fields))
                fig = explanation.as_pyplot_figure()
                lime_meta_filename = f"{uuid4()}_metadata_lime.jpg"
                lime_meta_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_meta_filename)
                fig.savefig(lime_meta_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                metadata_result['lime_filename'] = lime_meta_filename
            except Exception as e:
                logger.error(f"Metadata LIME failed: {e}")
                metadata_result['lime_filename'] = None

        return render_template('diagnosis.html', metadata_result=metadata_result, metadata_fields=metadata_fields,clinical_insights=clinical_insights_metadata)
    except Exception as e:
        logger.error(f"Error in /predict_metadata_form: {e}")
        return render_template('diagnosis.html', error=f"Error processing metadata: {e}", metadata_fields=metadata_fields)

@app.route('/api/health', methods=['GET'])
def api_health():
    """Report backend readiness and which configured checkpoints loaded."""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'models': {kind: list(loaded.keys()) for kind, loaded in preloaded_models.items()},
        'clinical_model_available': os.path.isfile(metadata_model_path) and os.path.isfile(SCALER_PATH),
    })


@app.route('/api/models', methods=['GET'])
def api_models():
    """Return only models that loaded successfully; never advertise missing weights."""
    return jsonify({kind: list(loaded.keys()) for kind, loaded in preloaded_models.items()})


def get_target_layer(model, layer_name):
    if layer_name == 'features':
        return model.features
    if layer_name == 'features.denseblock4':
        return model.features.denseblock4
    if layer_name == 'layer4':
        return model.layer4
    if layer_name == 'Mixed_7c':
        return model.Mixed_7c
    return list(model.children())[-1]


@app.route('/api/predict/image', methods=['POST'])
def api_predict_image():
    """Run the trained model ensemble and return its real predictions and explanations."""
    if 'user_id' not in session:
        return jsonify({'error': 'Please sign in before running image analysis.'}), 401
    scan_type = request.form.get('scan_type', '').lower()
    upload = request.files.get('image')
    if scan_type not in ('ct', 'xray'):
        return jsonify({'error': 'scan_type must be ct or xray.'}), 400
    if upload is None or not upload.filename:
        return jsonify({'error': 'Select an image to analyze.'}), 400
    try:
        filename = f'{uuid4()}.jpg'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            Image.open(upload.stream).convert('RGB').save(image_path, format='JPEG', quality=95)
        except Exception:
            return jsonify({'error': 'The selected file is not a supported image.'}), 400
        model_dict = model_paths[scan_type]
        label_names = class_names[scan_type]
        predictions = []
        probability_tensors = []
        for model_name, (_, _, layer_name) in model_dict.items():
            model = preloaded_models.get(scan_type, {}).get(model_name)
            if model is None:
                continue
            transform = model_transforms[model_name]
            try:
                result = enhanced_prediction_with_explanations(
                    image_path, model, model_name, transform,
                    get_target_layer(model, layer_name), label_names,
                    scan_type, lime_num_samples=50
                )
                if 'error' in result:
                    logger.error('API prediction failed for %s: %s', model_name, result['error'])
                    continue
                predictions.append({
                    'model': result['name'],
                    'label': result['label'],
                    'confidence': result['confidence'],
                    'probabilities': result['class_probs'],
                    'gradcamUrl': url_for('serve_uploaded_image', filename=result['gradcam_filename']) if result.get('gradcam_filename') else None,
                    'limeUrl': url_for('serve_uploaded_image', filename=result['lime_filename']) if result.get('lime_filename') else None,
                })
                tensor, _ = preprocess_image(image_path, transform)
                with torch.no_grad():
                    output = model(tensor.to(device))
                    if isinstance(output, tuple):
                        output = output[0]
                    probability_tensors.append(torch.softmax(output, dim=1).squeeze(0))
            except Exception as model_error:
                logger.exception('Model %s failed during API inference', model_name)
        if not probability_tensors:
            return jsonify({'error': 'No image models loaded successfully for this modality.'}), 503
        avg_probs = torch.stack(probability_tensors).mean(dim=0)
        confidence, prediction = torch.max(avg_probs, 0)
        consensus = label_names[prediction.item()]
        consensus_probs = {label_names[i]: float(avg_probs[i].item()) for i in range(len(label_names))}
        insights = generate_clinical_insights(consensus, float(confidence.item()), scan_type)
        return jsonify({
            'modality': scan_type,
            'model': 'Ensemble',
            'prediction': consensus,
            'confidence': float(confidence.item()),
            'probabilities': consensus_probs,
            'models': predictions,
            'imageUrl': url_for('serve_uploaded_image', filename=filename),
            'clinicalInsights': insights,
        })
    except Exception as error:
        logger.exception('Image API inference failed')
        return jsonify({'error': f'Unable to analyze image: {error}'}), 500


@app.route('/api/predict/clinical', methods=['POST'])
def api_predict_clinical():
    """Run the existing clinical MLP/scaler on submitted symptom fields."""
    try:
        payload = request.get_json(silent=True) or {}
        form_data = {field: payload.get(field) for field in metadata_fields}
        missing = [field for field, value in form_data.items() if value is None or str(value).strip() == '']
        if missing:
            return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400
        input_tensor = preprocess_metadata(form_data)
        metadata_model = load_metadata_model(metadata_model_path).to(device)
        with torch.no_grad():
            output = metadata_model(input_tensor.to(device))
            probs = torch.softmax(output, dim=1)[0]
        cancer_probability = float(probs[1].item())
        label = 'YES' if cancer_probability >= 0.5 else 'NO'
        confidence = cancer_probability if label == 'YES' else 1.0 - cancer_probability
        lime_url = None
        shap_url = None
        shap_description = None
        if LIME_AVAILABLE:
            try:
                explainer = LimeTabularExplainer(
                    training_data=np.random.randn(100, len(metadata_fields)),
                    feature_names=metadata_fields,
                    class_names=['No Cancer', 'Cancer'], mode='classification'
                )
                explanation = explainer.explain_instance(
                    input_tensor.cpu().numpy()[0],
                    make_predict_fn_tabular(metadata_model, device=device),
                    num_features=len(metadata_fields)
                )
                lime_filename = f'{uuid4()}_metadata_lime.jpg'
                lime_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_filename)
                figure = explanation.as_pyplot_figure()
                figure.savefig(lime_path, dpi=150, bbox_inches='tight')
                plt.close(figure)
                lime_url = url_for('serve_uploaded_image', filename=lime_filename)
            except Exception:
                logger.exception('Clinical LIME explanation failed')
        try:
            import shap

            class ProbabilityModel(nn.Module):
                def __init__(self, wrapped):
                    super().__init__()
                    self.wrapped = wrapped
                def forward(self, inputs):
                    return torch.softmax(self.wrapped(inputs), dim=1)

            # The zero vector is the mean baseline after the existing StandardScaler.
            probability_model = ProbabilityModel(metadata_model).eval()
            baseline = torch.zeros((1, len(metadata_fields)), dtype=torch.float32, device=device)
            shap_explainer = shap.GradientExplainer(probability_model, baseline)
            values = shap_explainer.shap_values(input_tensor.to(device))
            if isinstance(values, list):
                positive_values = np.asarray(values[1]).reshape(-1, len(metadata_fields))[0]
            else:
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[:, :, 1] if values.shape[-1] == 2 else values[1]
                positive_values = values.reshape(-1, len(metadata_fields))[0]
            order = np.argsort(np.abs(positive_values))[-12:]
            figure, axis = plt.subplots(figsize=(9, 5.2))
            ordered_values = positive_values[order]
            colors = ['#168b8d' if value >= 0 else '#c87570' for value in ordered_values]
            axis.barh([metadata_fields[i] for i in order], ordered_values, color=colors)
            axis.axvline(0, color='#718596', linewidth=0.8)
            axis.set_xlabel('SHAP contribution to model YES probability')
            axis.set_title('SHAP explanation for this clinical entry')
            axis.grid(axis='x', alpha=0.18)
            figure.tight_layout()
            shap_filename = f'{uuid4()}_clinical_shap.png'
            figure.savefig(os.path.join(app.config['UPLOAD_FOLDER'], shap_filename), dpi=150, bbox_inches='tight')
            plt.close(figure)
            shap_url = url_for('serve_uploaded_image', filename=shap_filename)
            shap_description = 'Local SHAP contributions relative to the model input mean baseline. Positive and negative values show how features move the model output; they are not medical causation.'
        except Exception:
            logger.exception('Clinical SHAP explanation failed')
        insights = generate_clinical_insights(label, confidence, 'metadata')
        return jsonify({
            'modality': 'metadata', 'model': 'MLP', 'prediction': label,
            'confidence': float(confidence),
            'probabilities': {'NO': float(probs[0].item()), 'YES': cancer_probability},
            'input': form_data, 'limeUrl': lime_url, 'shapUrl': shap_url,
            'shapDescription': shap_description, 'clinicalInsights': insights,
        })
    except Exception as error:
        logger.exception('Clinical API inference failed')
        return jsonify({'error': f'Unable to analyze clinical data: {error}'}), 500


@app.route('/api/predict/csv', methods=['POST'])
def api_predict_csv():
    """Predict a clinical CSV cohort and return cohort-level SHAP and LIME charts."""
    try:
        import pandas as pd
        import shap
        upload = request.files.get('file')
        if not upload or not upload.filename:
            return jsonify({'error': 'Choose a CSV file to analyze.'}), 400
        frame = pd.read_csv(upload)
        missing = [field for field in metadata_fields if field not in frame.columns]
        if missing:
            return jsonify({'error': f"CSV is missing required columns: {', '.join(missing)}"}), 400
        if frame.empty:
            return jsonify({'error': 'The CSV has no data rows.'}), 400
        if len(frame) > 5000:
            return jsonify({'error': 'CSV uploads are limited to 5,000 rows.'}), 400

        model = load_metadata_model(metadata_model_path).to(device).eval()
        row_features = []
        for _, row in frame.iterrows():
            values = {}
            for field in metadata_fields:
                value = row[field]
                if pd.isna(value):
                    return jsonify({'error': f'CSV has an empty value in {field}.'}), 400
                if field == 'GENDER':
                    values[field] = 'M' if str(value).strip().lower() in ('m', 'male', '1') else 'F'
                elif field == 'AGE':
                    try:
                        values[field] = float(value)
                    except (TypeError, ValueError):
                        return jsonify({'error': 'AGE must contain numeric values.'}), 400
                else:
                    values[field] = 'Yes' if str(value).strip().lower() in ('2', 'yes', 'true', 'y') else 'No'
            row_features.append(preprocess_metadata(values).numpy()[0])

        features = np.asarray(row_features, dtype=np.float32)
        tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()
        labels = ['YES' if score >= 0.5 else 'NO' for score in probabilities[:, 1]]

        # Use only rows from the uploaded cohort as the explainer background.
        # Limit SHAP work so larger cohort predictions stay responsive.
        explain_count = min(len(features), 64)
        explain_rows = features[:explain_count]
        background = features[np.linspace(0, len(features) - 1, min(len(features), 32), dtype=int)]
        class ProbabilityModel(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def forward(self, inputs):
                return torch.softmax(self.wrapped(inputs), dim=1)
        explainer_model = ProbabilityModel(model).eval()
        explainer = shap.GradientExplainer(
            explainer_model,
            torch.as_tensor(background, dtype=torch.float32, device=device),
        )
        shap_values = explainer.shap_values(torch.as_tensor(explain_rows, dtype=torch.float32, device=device))
        if isinstance(shap_values, list):
            positive_values = np.asarray(shap_values[1])
        else:
            positive_values = np.asarray(shap_values)
            if positive_values.ndim == 3:
                positive_values = positive_values[:, :, 1]
        positive_values = positive_values.reshape(explain_count, len(metadata_fields))
        importance = np.mean(np.abs(positive_values), axis=0)
        order = np.argsort(importance)[-12:]
        figure, axis = plt.subplots(figsize=(9, 5.2))
        axis.barh([metadata_fields[i] for i in order], importance[order], color='#168f91')
        axis.set_xlabel('Mean absolute SHAP value (YES class probability)')
        axis.set_title(f'Clinical cohort feature importance · {explain_count} of {len(features)} rows explained')
        axis.grid(axis='x', alpha=0.2)
        figure.tight_layout()
        chart_name = f'{uuid4()}_clinical_shap.png'
        figure.savefig(os.path.join(app.config['UPLOAD_FOLDER'], chart_name), dpi=150, bbox_inches='tight')
        plt.close(figure)
        lime_url = None
        lime_description = None
        if LIME_AVAILABLE:
            try:
                lime_count = min(len(features), 16)
                lime_indices = np.linspace(0, len(features) - 1, lime_count, dtype=int)
                lime_explainer = LimeTabularExplainer(
                    training_data=background,
                    feature_names=metadata_fields,
                    class_names=['No Cancer', 'Cancer'],
                    mode='classification',
                )
                predict_fn = make_predict_fn_tabular(model, device=device)
                local_importances = []
                for row_index in lime_indices:
                    local = lime_explainer.explain_instance(
                        features[row_index], predict_fn, labels=(1,),
                        num_features=len(metadata_fields), num_samples=512,
                    )
                    weights = np.zeros(len(metadata_fields), dtype=np.float32)
                    for feature_index, weight in local.local_exp.get(1, []):
                        weights[int(feature_index)] = float(weight)
                    local_importances.append(np.abs(weights))
                lime_importance = np.mean(local_importances, axis=0)
                lime_order = np.argsort(lime_importance)[-12:]
                lime_figure, lime_axis = plt.subplots(figsize=(9, 5.2))
                lime_axis.barh([metadata_fields[i] for i in lime_order], lime_importance[lime_order], color='#6d91c5')
                lime_axis.set_xlabel('Mean absolute LIME weight for the model Cancer class')
                lime_axis.set_title(f'LIME cohort summary · {lime_count} representative rows')
                lime_axis.grid(axis='x', alpha=0.18)
                lime_figure.tight_layout()
                lime_filename = f'{uuid4()}_clinical_lime.png'
                lime_figure.savefig(os.path.join(app.config['UPLOAD_FOLDER'], lime_filename), dpi=150, bbox_inches='tight')
                plt.close(lime_figure)
                lime_url = url_for('serve_uploaded_image', filename=lime_filename)
                lime_description = f'Mean absolute local LIME weights aggregated across {lime_count} representative rows. This summary describes model behavior, not medical causation.'
            except Exception:
                logger.exception('CSV LIME explanation failed')
        return jsonify({
            'row_count': len(features),
            'explained_rows': explain_count,
            'shap_url': url_for('serve_uploaded_image', filename=chart_name),
            'chart_description': 'Cohort-level mean absolute SHAP values for the model YES-class probability. These explain model behavior, not medical causation.',
            'lime_url': lime_url,
            'lime_description': lime_description,
            'predictions': [{'row': i + 1, 'prediction': labels[i], 'probability_yes': float(probabilities[i, 1])} for i in range(len(features))],
            'class_counts': {'YES': labels.count('YES'), 'NO': labels.count('NO')},
        })
    except ImportError:
        return jsonify({'error': 'SHAP is not installed in the active backend environment.'}), 503
    except Exception as error:
        logger.exception('CSV SHAP inference failed')
        return jsonify({'error': f'Unable to analyze CSV: {error}'}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    from chatbot.service import answer
    payload = request.get_json(silent=True) or {}
    message = payload.get('message', '')
    if not isinstance(message, str) or len(message) > 2000:
        return jsonify({'error': 'Message must be text no longer than 2,000 characters.'}), 400
    context = payload.get('context')
    history = payload.get('history')
    result = answer(message, context, history)
    result['sources'] = [{'title': source['title'], 'url': source['url']} for source in result.get('sources', [])]
    result['conversation_id'] = payload.get('conversation_id') or str(uuid4())
    return jsonify(result)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnosis():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    logger.debug("Entered /diagnosis route")
    if request.method == 'POST':
        scan_type = request.form.get('scan_type')

        if scan_type == 'metadata':
            if 'csv_file' not in request.files:
                return render_template('diagnosis.html', error="No CSV file uploaded", metadata_fields=metadata_fields)
            csv_file = request.files['csv_file']
            if csv_file.filename == '':
                return render_template('diagnosis.html', error="No CSV file selected", metadata_fields=metadata_fields)
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if not all(col in df.columns for col in metadata_fields):
                    return render_template('diagnosis.html', error="CSV must contain all required columns", metadata_fields=metadata_fields)
                form_data = {}
                for field in metadata_fields:
                    val = df[field].iloc[0]
                    if field == 'GENDER':
                        form_data[field] = 'M' if str(val).strip().upper() == 'M' else 'F'
                    elif field == 'AGE':
                        form_data[field] = int(val)
                    else:
                        if str(val).strip() in ['2', 'Yes', 'yes', 'TRUE', 'True', 'true', '1']:
                            form_data[field] = 'Yes'
                        else:
                            form_data[field] = 'No'
                input_tensor = preprocess_metadata(form_data)
                metadata_model = load_metadata_model(metadata_model_path)
                metadata_model = metadata_model.to(device)
                with torch.no_grad():
                    output = metadata_model(input_tensor.to(device))
                    probs = torch.softmax(output, dim=1)
                    raw_output = probs[0][1].item()
                    predicted_label = "YES" if raw_output >= 0.5 else "NO"
                    confidence = raw_output if raw_output >= 0.5 else 1.0 - raw_output
                metadata_result = {'label': predicted_label, 'confidence': confidence, 'input': form_data, 'raw_output': raw_output}


                # ---------- LIME for CSV metadata ----------
                if LIME_AVAILABLE:
                    try:
                        lime_explainer = LimeTabularExplainer(
                            training_data=np.random.randn(100, len(metadata_fields)),
                            feature_names=metadata_fields,
                            class_names=['No Cancer', 'Cancer'],
                            mode='classification'
                        )

                        features_array = input_tensor.cpu().numpy()
                        explanation = lime_explainer.explain_instance(
                           features_array[0],
                           make_predict_fn_tabular(metadata_model, device=device),
                           num_features=len(metadata_fields)
                        )

                        fig = explanation.as_pyplot_figure()
                        lime_meta_filename = f"{uuid4()}_metadata_lime.jpg"
                        lime_meta_path = os.path.join(app.config['UPLOAD_FOLDER'], lime_meta_filename)
                        fig.savefig(lime_meta_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)

                        metadata_result['lime_filename'] = lime_meta_filename

                    except Exception as e:
                       logger.error(f"CSV Metadata LIME failed: {e}")
                       metadata_result['lime_filename'] = None
# -----------------------------------------

                
        
        # Generate clinical insights for CSV metadata
                clinical_insights_metadata = generate_clinical_insights(
                    consensus_label=predicted_label,
                    consensus_conf=confidence,
                    scan_type='metadata',
                    metadata_result=metadata_result
               )
        
                return render_template('diagnosis.html', 
                                     metadata_result=metadata_result, 
                                     metadata_fields=metadata_fields,
                                     clinical_insights=clinical_insights_metadata)
            except Exception as e:
                logger.error(f"Error processing CSV: {e}")
                return render_template('diagnosis.html', error=f"Error processing CSV: {e}", metadata_fields=metadata_fields)

        elif scan_type == 'metadata_form':
            return render_template('diagnosis.html', metadata_fields=metadata_fields)

        if 'image' not in request.files:
            return render_template('diagnosis.html', error="No image uploaded", metadata_fields=metadata_fields)
        file = request.files['image']
        if file.filename == '':
            return render_template('diagnosis.html', error="No image selected", metadata_fields=metadata_fields)
        try:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid4()}.jpg")
            file.save(image_path)

            results = []
            probabilities = []
            model_dict = model_paths.get(scan_type)
            if model_dict is None:
                return render_template('diagnosis.html', error="Invalid scan type", metadata_fields=metadata_fields)
            class_names_list = class_names[scan_type]

            for name, (_, _, target_layer_name) in model_dict.items():
                transform = model_transforms.get(name, transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]))
                model = preloaded_models.get(scan_type, {}).get(name)
                if model is None:
                    logger.error(f"Preloaded model {name} not found for {scan_type}")
                    continue

                try:
                    if target_layer_name == 'features':
                        target_layer = model.features
                    elif target_layer_name == 'features.denseblock4':
                        target_layer = model.features.denseblock4
                    elif target_layer_name == 'layer4':
                        target_layer = model.layer4
                    elif target_layer_name == 'Mixed_7c':
                        target_layer = model.Mixed_7c
                    else:
                        target_layer = list(model.children())[-1]
                except Exception:
                    target_layer = list(model.children())[-1]

                model_result = enhanced_prediction_with_explanations(
                    image_path, model, name, transform, target_layer, class_names_list,
                    scan_type, lime_num_samples=DEFAULT_LIME_NUM_SAMPLES
                )

                if 'error' not in model_result:
                    results.append({
                        'name': model_result.get('name'),
                        'label': model_result.get('label'),
                        'confidence': model_result.get('confidence'),
                        'class_probs': model_result.get('class_probs'),
                        'gradcam_filename': model_result.get('gradcam_filename') or '',
                        'overlay_filename': model_result.get('overlay_filename') or '',
                        'lime_filename': model_result.get('lime_filename') or ''
                    })
                    input_tensor, _ = preprocess_image(image_path, transform)
                    with torch.no_grad():
                        out = model(input_tensor.to(device))
                        if isinstance(out, tuple):
                            out = out[0]
                        probs = torch.softmax(out, dim=1).squeeze(0)
                        probabilities.append(probs)
                else:
                    logger.error(f"Model {name} failed: {model_result.get('error')}")

            if probabilities:
                avg_probs = torch.stack(probabilities).mean(dim=0)
                consensus_conf, consensus_pred = torch.max(avg_probs, 0)
                consensus_label = class_names_list[consensus_pred.item()]
                consensus_class_probs = {class_names_list[i]: avg_probs[i].item() for i in range(len(class_names_list))}
                
                # Generate clinical insights
                clinical_insights = generate_clinical_insights(
                    consensus_label=consensus_label,
                    consensus_conf=consensus_conf.item() if hasattr(consensus_conf, 'item') else float(consensus_conf),
                    scan_type=scan_type,
                    metadata_result=None
                )
            else:
                consensus_label = "Error"
                consensus_conf = 0.0
                consensus_class_probs = {}
                clinical_insights = None

            return render_template('diagnosis.html',
                                   results=results,
                                   consensus_label=consensus_label,
                                   consensus_conf=consensus_conf.item() if hasattr(consensus_conf, 'item') else float(consensus_conf),
                                   consensus_class_probs=consensus_class_probs,
                                   original_image_filename=os.path.basename(image_path),
                                   metadata_fields=metadata_fields,
                                   lime_available=LIME_AVAILABLE,
                                   clinical_insights=clinical_insights)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return render_template('diagnosis.html', error=f"Error processing image: {e}", metadata_fields=metadata_fields)

    return render_template('diagnosis.html', metadata_fields=metadata_fields)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/moreinfo')
def moreinfo():
    return render_template('moreinfo.html')

@app.route('/documentation')
def serve_documentation():
    try:
        return send_file(os.path.join('static', 'LungVision_AI_documentation.pdf'))
    except Exception:
        abort(404)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.json
        full_name = data.get('fullName')
        email = data.get('email')
        password = data.get('password')
        role = data.get('role')
        
        if not all([full_name, email, password, role]):
            return jsonify({'success': False, 'error': 'Please fill all fields.'}), 400
        
        # Validate password
        if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'\d', password) or not re.search(r'[!@#$%^&*]', password):
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters with 1 uppercase, 1 number, and 1 special character.'}), 400
        
        # Check if email exists
        if User.query.filter_by(email=email).first():
            return jsonify({'success': False, 'error': 'Email already registered.'}), 400
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(
            full_name=full_name,
            email=email,
            password_hash=hashed_password,
            role=role
        )
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True})
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('diagnosis'))
    
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        if not user:
            record_login_audit('login', 'failure')
            return jsonify({'success': False, 'error': 'account_not_found'}), 400
        if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash):
            record_login_audit('login', 'failure')
            return jsonify({'success': False, 'error': 'incorrect_password'}), 400
        
        session['user_id'] = user.id
        record_login_audit('login', 'success', user.id)
        return jsonify({'success': True})
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/forgot_password')
def forgot_password():
    # Placeholder for password reset (implement as needed, e.g., email reset link)
    return "Password reset functionality coming soon."

@app.route('/get_explanation/<explanation_type>/<filename>')
def get_explanation_details(explanation_type, filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'Explanation file not found'}), 404

# Load Gemini API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Validate API key
if GEMINI_AVAILABLE:
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not configured. The assistant will show source links but cannot generate answers.")
    else:
        logger.info("Gemini API key is configured")


@app.route('/chat', methods=['POST'])
def chat():
    payload = request.get_json(silent=True) or {}
    from chatbot.service import answer
    result = answer(payload.get('message', ''), payload.get('context'))
    return jsonify({'response': result['answer'], 'answer': result['answer'], 'sources': result['sources']})
# Preload models at module level (works with both Flask dev server and Gunicorn)
logger.info("=== Starting model preloading ===")
torch.set_num_threads(4)
preload_models()
logger.info("=== Model preloading complete ===")

if __name__ == '__main__':
    if not LIME_AVAILABLE:
        logger.warning("LIME not available. Install with: pip install lime")
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available. Install with: pip install google-genai")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
