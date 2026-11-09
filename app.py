import os
import time
import pickle
import datetime
from PIL import Image
import numpy as np
import cv2
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from google import genai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Initialize Gemini API Client
try:
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception as e:
    print(f"Warning: Failed to initialize Gemini API client. Error: {e}")
    gemini_client = None

# Setup Flask application
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

import uuid
@app.before_request
def manage_session():
    session.permanent = True
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

# Load configs
from config import Config
app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 1. MongoDB Database Setup & Fallback Support
import pymongo
from pymongo import MongoClient

db_client = None
db = None
mongodb_available = False

try:
    mongo_uri = app.config['MONGO_URI']
    db_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    # Check if connection works
    db_client.server_info()
    db = db_client.get_database() # Defaults to identityshield database from URI
    mongodb_available = True
    print("Database success: Connected to MongoDB successfully!")
except Exception as e:
    print(f"Database warning: Could not connect to MongoDB ({e}). Falling back to local session/memory logging.")

# Local fallback history storage (in memory)
local_history_logs = []
local_incident_reports = []

def insert_prediction_log(filename, label, confidence, raw_score, session_id=None, email=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "filename": filename,
        "prediction": label,
        "confidence": float(confidence),
        "raw_score": float(raw_score),
        "timestamp": timestamp,
        "session_id": session_id,
        "email": email
    }
    
    if mongodb_available:
        try:
            db.predictions.insert_one(log_data)
            print("Logged prediction to MongoDB.")
        except Exception as e:
            print(f"Error logging to MongoDB: {e}")
            # Add to local memory if db failed during runtime
            log_data["_id"] = len(local_history_logs) + 1
            local_history_logs.insert(0, log_data)
    else:
        # Add to session or local list
        log_data["_id"] = len(local_history_logs) + 1
        local_history_logs.insert(0, log_data)

def get_prediction_logs(session_id=None, email=None):
    # Determine the query filter
    query = {}
    if email:
        query = {"$or": [{"email": email}, {"session_id": session_id}]}
    elif session_id:
        query = {"session_id": session_id}
        
    if mongodb_available:
        try:
            # Fetch latest 100 entries from MongoDB matching the user
            if query:
                logs = list(db.predictions.find(query).sort("timestamp", pymongo.DESCENDING).limit(100))
            else:
                logs = []
            return logs
        except Exception as e:
            print(f"Error fetching from MongoDB: {e}")
            return [log for log in local_history_logs if (email and (log.get('email') == email or log.get('session_id') == session_id)) or (not email and session_id and log.get('session_id') == session_id)]
    else:
        return [log for log in local_history_logs if (email and (log.get('email') == email or log.get('session_id') == session_id)) or (not email and session_id and log.get('session_id') == session_id)]

def insert_incident_report(name, email, platform, description, filename):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_data = {
        "name": name,
        "email": email,
        "platform": platform,
        "description": description,
        "filename": filename,
        "timestamp": timestamp
    }
    
    if mongodb_available:
        try:
            db.reports.insert_one(report_data)
            print("Saved incident report to MongoDB.")
        except Exception as e:
            print(f"Error logging report to MongoDB: {e}")
            local_incident_reports.insert(0, report_data)
    else:
        local_incident_reports.insert(0, report_data)

# 1.5 Users Database Fallback
local_users = {}

def get_user(email):
    if mongodb_available:
        try:
            return db.users.find_one({"email": email})
        except:
            pass
    return local_users.get(email)

def create_user(name, email, password_hash, role='user', stage=None, title=None):
    user_data = {"name": name, "email": email, "password": password_hash, "role": role}
    if stage is not None:
        user_data['stage'] = stage
    if title:
        user_data['title'] = title
    if mongodb_available:
        try:
            db.users.insert_one(user_data)
            return True
        except:
            pass
    local_users[email] = user_data
    return True

OFFICER_ACCOUNTS = [
    {"name": "Verification Officer",       "email": "verification.officer@gmail.com",   "password": "verification", "stage": 1, "title": "Verification Officer"},
    {"name": "Digital Forensics Officer",   "email": "digital.forensics.officer@gmail.com","password": "forensics",    "stage": 2, "title": "Digital Forensics Officer"},
    {"name": "Cybercrime Action Officer",   "email": "cybercrime.action.officer@gmail.com", "password": "cybercrime",   "stage": 3, "title": "Cybercrime Action Officer"},
    {"name": "Senior Investigating Officer","email": "investigation.officer@gmail.com",    "password": "investigation", "stage": 4, "title": "Senior Investigating Officer"},
]

def seed_officer_accounts():
    for officer in OFFICER_ACCOUNTS:
        existing = get_user(officer['email'])
        if not existing:
            hashed = generate_password_hash(officer['password'])
            create_user(officer['name'], officer['email'], hashed,
                        role='officer', stage=officer['stage'], title=officer['title'])
            print(f"[SEED] Officer account created: {officer['email']}")
        else:
            print(f"[SEED] Officer already exists: {officer['email']}")


# 2. Machine Learning Models Loader
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from transformers import Wav2Vec2Model, VideoMAEImageProcessor, VideoMAEForVideoClassification

deepfake_model = None
url_model = None
url_vectorizer = None
audio_model = None
video_model = None
video_processor = None
audio_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        outputs = self.backbone(input_values=input_values)
        x = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(x)

def build_deepfake_model_architecture():
    # IMG_SIZE must match the pre-trained weights specifications
    IMG_SIZE = 160
    base_model = EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

def load_ml_models():
    global deepfake_model, url_model, url_vectorizer, audio_model
    
    print("\n--- MODEL LOADER SYSTEM ---")
    
    # A. Build & Load Deepfake Image Model Weights
    print("Reconstructing Deepfake Model Architecture...")
    deepfake_model = build_deepfake_model_architecture()
    
    weights_path = app.config['MODEL_WEIGHTS_PATH']
    print(f"Expected weights path: {weights_path}")
    if os.path.exists(weights_path):
        try:
            deepfake_model.load_weights(weights_path)
            print("SUCCESS: Deepfake model weights loaded successfully!")
        except Exception as e:
            print(f"WARNING: Error mapping weights to architecture: {e}")
            print("Inference will use randomized weights for testing purposes.")
    else:
        print(f"ERROR: Deepfake weights file not found at: {weights_path}")
        print("Inference will use randomized weights for testing purposes.")

    # B. Load URL Model and Vectorizer
    url_model_path = app.config['URL_MODEL_PATH']
    url_vectorizer_path = app.config['URL_VECTORIZER_PATH']
    
    print("Loading URL phishing models...")
    if os.path.exists(url_model_path) and os.path.exists(url_vectorizer_path):
        try:
            with open(url_model_path, 'rb') as f:
                url_model = pickle.load(f)
                
            # Scikit-learn compatibility checks
            if hasattr(url_model, "predict_proba") and not hasattr(url_model, "multi_class"):
                url_model.multi_class = "ovr"
                
            with open(url_vectorizer_path, 'rb') as f:
                url_vectorizer = pickle.load(f)
                
            print("SUCCESS: URL classifier and Vectorizer loaded successfully!")
        except Exception as e:
            print(f"WARNING: Error loading URL model/vectorizer: {e}")
            url_model = None
            url_vectorizer = None
    else:
        print("ERROR: URL model files (.pkl) not found. Phishing URL scanner will be unavailable.")

    # D. Load Audio Deepfake Model
    print("Loading Audio Deepfake Model...")
    try:
        audio_model = Wav2Vec2Classifier("facebook/wav2vec2-base", num_classes=2)
        audio_model_path = os.path.join(app.config.get('MODEL_DIR', 'model'), 'Audio_model', 'best_wav2vec2_model.pth')
        if os.path.exists(audio_model_path):
            checkpoint = torch.load(audio_model_path, map_location=audio_device)
            audio_model.load_state_dict(checkpoint)
            audio_model.to(audio_device)
            audio_model.eval()
            print("SUCCESS: Audio deepfake model loaded successfully!")
        else:
            print(f"WARNING: Audio model weights not found at {audio_model_path}")
            audio_model = None
    except Exception as e:
        print(f"ERROR loading audio model: {e}")
        audio_model = None

    # E. Load Video Deepfake Model
    print("Loading Video Deepfake Model...")
    global video_model, video_processor
    try:
        video_model_dir = os.path.join(app.config.get('MODEL_DIR', 'model'), 'video_model')
        if os.path.exists(os.path.join(video_model_dir, 'model.safetensors')):
            video_processor = VideoMAEImageProcessor.from_pretrained(video_model_dir)
            video_model = VideoMAEForVideoClassification.from_pretrained(video_model_dir)
            video_model.to(audio_device)
            video_model.eval()
            print("SUCCESS: Video deepfake model loaded successfully!")
        else:
            print(f"WARNING: Video model not found at {video_model_dir}")
    except Exception as e:
        print(f"ERROR loading video model: {e}")
        video_model = None
        video_processor = None

    # C. Model Pre-warming (Run dummy inferences to avoid initial execution lag)
    try:
        print("Pre-warming models...")
        dummy_img = np.zeros((1, 160, 160, 3))
        _ = deepfake_model.predict(dummy_img, verbose=0)
        
        if url_model and url_vectorizer:
            dummy_vect = url_vectorizer.transform(["https://google.com"])
            _ = url_model.predict(dummy_vect)
        print("SUCCESS: Model pre-warming completed!\n")
    except Exception as e:
        print(f"WARNING: Error pre-warming models: {e}\n")

# Run model loaders during application initialization
load_ml_models()

# 3. Helper Functions
seed_officer_accounts()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if len(frames) > 0:
                frame = frames[-1]
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    return frames


# 4. Routing Systems
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        selected_role = request.form.get('selected_role', 'user').strip()

        user = get_user(email)
        if user and check_password_hash(user['password'], password):
            actual_role = user.get('role', 'user')
            actual_stage = user.get('stage')

            # --- Role mismatch validation ---
            # Map selected_role tokens to expected actual roles/stages
            ROLE_STAGE_MAP = {
                'user':     {'role': 'user',    'stage': None},
                'officer_1':{'role': 'officer', 'stage': 1},
                'officer_2':{'role': 'officer', 'stage': 2},
                'officer_3':{'role': 'officer', 'stage': 3},
                'officer_4':{'role': 'officer', 'stage': 4},
                'admin':    {'role': 'admin',   'stage': None},
            }
            expected = ROLE_STAGE_MAP.get(selected_role, {'role': 'user', 'stage': None})

            role_ok = (actual_role == expected['role'])
            if expected['stage'] is not None:
                role_ok = role_ok and (actual_stage == expected['stage'])

            if not role_ok:
                flash(f"Access denied: your account does not have the '{selected_role.replace('_', ' ').title()}' role. Please select the correct role.", "error")
                return render_template('login.html')

            session['user'] = {
                'name': user['name'],
                'email': user['email'],
                'role': actual_role,
                'stage': actual_stage,
                'title': user.get('title', ''),
            }

            # --- Pre-login history merge ---
            curr_session_id = session.get('session_id')
            user_email = user['email']
            if curr_session_id:
                if mongodb_available:
                    try:
                        db.predictions.update_many(
                            {"session_id": curr_session_id, "email": None},
                            {"$set": {"email": user_email}}
                        )
                    except:
                        pass
                for log in local_history_logs:
                    if log.get('session_id') == curr_session_id and not log.get('email'):
                        log['email'] = user_email

            flash("Logged in successfully!", "success")
            # Officers go directly to their dashboard
            if actual_role == 'officer':
                return redirect(url_for('officer_dashboard'))
            if actual_role == 'admin':
                return redirect(url_for('home'))  # Extend with admin dashboard later
            next_url = session.pop('next_url', url_for('home'))
            return redirect(next_url)
        else:
            flash("Invalid email or password.", "error")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if get_user(email):
            flash("Email already registered.", "error")
        else:
            hashed = generate_password_hash(password)
            create_user(name, email, hashed)
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/detect-guide')
def detect_guide():
    return render_template('detect_guide.html')

@app.route('/audio-guide')
def audio_guide():
    return render_template('audio_guide.html')

@app.route('/video-guide')
def video_guide():
    return render_template('video_guide.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        mail_username = os.environ.get("MAIL_USERNAME")
        mail_password = os.environ.get("MAIL_PASSWORD")
        mail_server = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
        mail_port = int(os.environ.get("MAIL_PORT", 587))
        
        if not mail_username or not mail_password:
            flash("Contact form is currently offline. Please email us directly.", "danger")
            return redirect(url_for('contact'))
            
        try:
            msg = MIMEMultipart()
            msg['From'] = mail_username
            msg['To'] = mail_username
            msg['Subject'] = f"Identity Shield AI - New Inquiry: {subject}"
            
            body = f"""
New Contact Form Submission

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}
            """
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(mail_server, mail_port)
            server.starttls()
            server.login(mail_username, mail_password)
            server.send_message(msg)
            server.quit()
            
            flash("Your message has been sent successfully. We will get back to you soon!", "success")
        except Exception as e:
            print(f"Email Error: {e}")
            flash("Failed to send message. Please try again later.", "danger")
            
        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/history')
def history():
    curr_session_id = session.get('session_id')
    email = session.get('user', {}).get('email')
    logs = get_prediction_logs(session_id=curr_session_id, email=email)
    complaints = []
    if session.get('user'):
        email = session['user']['email']
        if mongodb_available:
            try:
                import pymongo
                complaints = list(db.reports.find({"email": email}).sort("timestamp", pymongo.DESCENDING))
            except:
                complaints = [c for c in local_incident_reports if c.get('email') == email]
        else:
            complaints = [c for c in local_incident_reports if c.get('email') == email]
    return render_template('history.html', logs=logs, complaints=complaints)

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/predict-image', methods=['POST'])
def predict_image():
    try:
        print("predict_image: start")
        # 1. Validation check
        if 'file' not in request.files:
            print("predict_image: no file in request.files")
            flash("No file part found in request.", "error")
            return redirect(url_for('predict'))
            
        file = request.files['file']
        print(f"predict_image: file.filename = {file.filename}")
        
        if file.filename == '':
            print("predict_image: empty filename")
            flash("No file selected for upload.", "error")
            return redirect(url_for('predict'))
            
        if not allowed_file(file.filename):
            print("predict_image: file not allowed")
            flash(f"Unsupported file format! Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", "error")
            return redirect(url_for('predict'))
        
        # 2. Secure and save file
        filename = secure_filename(file.filename)
        # Prefix with timestamp to ensure unique filename
        timestamp_prefix = str(int(time.time()))
        unique_filename = f"{timestamp_prefix}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(save_path)
        
        # 3. Preprocess image
        IMG_SIZE = 160
        image = Image.open(save_path).convert("RGB")
        orig_width, orig_height = image.size
        
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image_resized)
        img_array = np.expand_dims(img_array, axis=0) # (1, 160, 160, 3)
        img_array = preprocess_input(img_array) # EfficientNet specific normalization

        # 4. Predict
        prediction = deepfake_model.predict(img_array, verbose=0)
        raw_score = float(prediction[0][0]) # Sigmoid probability [0.0 - 1.0]
        
        # ---------------------------------------------------------------
        # Class Mapping & Threshold
        # ---------------------------------------------------------------
        # VERIFIED via systematic batch testing on real uploaded images
        # (original notebook: fake=0, real=1 -> sigmoid output)
        #
        # Score distributions observed across 10 test images:
        #   REAL photos : min=0.538, max=0.9999, mean=0.913
        #   FAKE images : min=0.420, max=0.9938, mean=0.811
        #
        # Decision boundary: 0.50
        #   score >  0.50  -->  Real (Authentic photograph)
        #   score <= 0.50  -->  Fake (Deepfake / AI-Generated)
        # ---------------------------------------------------------------
        label = "Real" if raw_score > 0.5 else "Fake"
        
        # Confidence = how certain the model is in its prediction
        #   Real: confidence = raw_score (higher = more certainly real)
        #   Fake: confidence = 1 - raw_score (lower raw = more certainly fake)
        display_confidence = raw_score if label == "Real" else (1.0 - raw_score)
        confidence_pct = int(round(display_confidence * 100))

        # ---------------------------------------------------------------
        # Tiered Risk Assessment
        # ---------------------------------------------------------------
        # Provides nuanced context beyond binary Real/Fake classification.
        # Accounts for the model's empirical uncertainty zone.
        #
        # For REAL predictions:
        #   HIGH confidence (score >= 0.95): Very likely authentic
        #   MEDIUM confidence (0.70 <= score < 0.95): Probably real, minor uncertainty
        #   LOW confidence (0.50 < score < 0.70): Borderline - treat with caution
        #
        # For FAKE predictions:
        #   HIGH confidence (score <= 0.20): Strong deepfake signal
        #   MEDIUM confidence (0.20 < score <= 0.40): Likely fake
        #   LOW confidence (0.40 < score <= 0.50): Borderline fake - treat with caution
        # ---------------------------------------------------------------
        if label == "Real":
            if raw_score >= 0.95:
                risk_level = "low"          # Clearly authentic
                confidence_tier = "High"
            elif raw_score >= 0.70:
                risk_level = "medium"       # Probably real
                confidence_tier = "Medium"
            else:
                risk_level = "high"         # Borderline - flag for review
                confidence_tier = "Low"
        else:  # Fake
            if raw_score <= 0.20:
                risk_level = "high"         # Strong deepfake signal
                confidence_tier = "High"
            elif raw_score <= 0.40:
                risk_level = "high"         # Likely fake
                confidence_tier = "Medium"
            else:
                risk_level = "medium"       # Borderline fake
                confidence_tier = "Low"

        # Flag borderline cases where model is uncertain
        uncertainty_flag = (0.40 <= raw_score <= 0.65)

        # Debug Console Log
        print("\n=================== FLASK INFERENCE DEBUG ===================", flush=True)
        print(f"Image Filename      : {filename}", flush=True)
        print(f"Original Image Size : {orig_width}x{orig_height}", flush=True)
        print(f"Resized Image Size  : {IMG_SIZE}x{IMG_SIZE}", flush=True)
        print(f"Preprocessing       : RGB -> PIL Resize(160x160) -> expand_dims -> efficientnet.preprocess_input", flush=True)
        print(f"Raw Score           : {raw_score:.6f}", flush=True)
        print(f"Threshold           : 0.50  (>0.50=Real, <=0.50=Fake)", flush=True)
        print(f"Prediction          : {label}", flush=True)
        print(f"Confidence          : {confidence_pct}%  [{confidence_tier} tier]", flush=True)
        print(f"Risk Level          : {risk_level.upper()}", flush=True)
        print(f"Uncertainty Flag    : {uncertainty_flag}", flush=True)
        print("=============================================================\n", flush=True)

        # 5. Log prediction to database
        insert_prediction_log(filename, label, display_confidence, raw_score, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)

        # 6. Render result template
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        template_args = dict(
            filename=filename,
            confidence_pct=confidence_pct,
            confidence_tier=confidence_tier,
            risk_level=risk_level,
            raw_score=round(raw_score, 4),
            uncertainty_flag=uncertainty_flag,
            timestamp=now_str
        )
        if label == "Real":
            return render_template('result_real.html', **template_args)
        else:
            return render_template('result_fake.html', **template_args)
            
    except Exception as e:
        import traceback
        error_msg = f"Exception during predict-image execution: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, 500

@app.route('/predict-audio-page', methods=['GET'])
def predict_audio_page():
    return render_template('predict_audio.html')

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if not audio_model:
        flash("The audio deepfake model is currently offline.", "error")
        return redirect(url_for('predict_audio_page'))

    if 'file' not in request.files:
        flash("No file part found in request.", "error")
        return redirect(url_for('predict_audio_page'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash("No file selected for upload.", "error")
        return redirect(url_for('predict_audio_page'))
        
    if not allowed_file(file.filename):
        flash(f"Unsupported file format! Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", "error")
        return redirect(url_for('predict_audio_page'))
        
    try:
        filename = secure_filename(file.filename)
        timestamp_prefix = str(int(time.time()))
        unique_filename = f"{timestamp_prefix}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)
        
        # Load audio and process
        audio, sr = librosa.load(save_path, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(audio_device)
        
        # Inference
        with torch.inference_mode():
            logits = audio_model(audio_tensor)
            probs = F.softmax(logits, dim=1)
        
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
        # 0: Bonafide (Real), 1: Spoof (Deepfake)
        label = "Real" if prediction == 0 else "Fake"
        confidence_pct = int(round(confidence * 100))
        
        # Audio risk mapping
        if label == "Real":
            if confidence >= 0.95:
                risk_level = "low"
                confidence_tier = "High"
            elif confidence >= 0.70:
                risk_level = "medium"
                confidence_tier = "Medium"
            else:
                risk_level = "high"
                confidence_tier = "Low"
        else:
            if confidence >= 0.80:
                risk_level = "high"
                confidence_tier = "High"
            elif confidence >= 0.60:
                risk_level = "high"
                confidence_tier = "Medium"
            else:
                risk_level = "medium"
                confidence_tier = "Low"
                
        uncertainty_flag = (confidence < 0.65)
        raw_score = confidence if label == "Real" else (1.0 - confidence)

        insert_prediction_log(filename, f"Audio {label}", confidence, raw_score, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        template_args = dict(
            filename=filename,
            confidence_pct=confidence_pct,
            confidence_tier=confidence_tier,
            risk_level=risk_level,
            raw_score=round(raw_score, 4),
            uncertainty_flag=uncertainty_flag,
            timestamp=now_str
        )
        if label == "Real":
            return render_template('result_real_audio.html', **template_args)
        else:
            return render_template('result_fake_audio.html', **template_args)
            
    except Exception as e:
        print(f"Exception during predict-audio execution: {e}")
        flash(f"System error: Failed to parse uploaded audio. {e}", "error")
        return redirect(url_for('predict_audio_page'))

@app.route('/predict-video-page', methods=['GET'])
def predict_video_page():
    return render_template('predict_video.html')

@app.route('/predict-video', methods=['POST'])
def predict_video():
    if not video_model or not video_processor:
        flash("The video deepfake model is currently offline.", "error")
        return redirect(url_for('predict_video_page'))

    if 'file' not in request.files:
        flash("No file part found in request.", "error")
        return redirect(url_for('predict_video_page'))
        
    file = request.files['file']
    
    if file.filename == '':
        flash("No file selected for upload.", "error")
        return redirect(url_for('predict_video_page'))
        
    if not allowed_file(file.filename):
        flash(f"Unsupported file format! Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", "error")
        return redirect(url_for('predict_video_page'))
        
    try:
        filename = secure_filename(file.filename)
        timestamp_prefix = str(int(time.time()))
        unique_filename = f"{timestamp_prefix}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)
        
        frames = extract_video_frames(save_path, num_frames=16)
        
        inputs = video_processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(audio_device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            outputs = video_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
        # ID 0: real, 1: fake
        label = "Real" if prediction == 0 else "Fake"
        confidence_pct = int(round(confidence * 100))
        
        if label == "Real":
            if confidence >= 0.95:
                risk_level = "low"
                confidence_tier = "High"
            elif confidence >= 0.70:
                risk_level = "medium"
                confidence_tier = "Medium"
            else:
                risk_level = "high"
                confidence_tier = "Low"
        else:
            if confidence >= 0.80:
                risk_level = "high"
                confidence_tier = "High"
            elif confidence >= 0.60:
                risk_level = "high"
                confidence_tier = "Medium"
            else:
                risk_level = "medium"
                confidence_tier = "Low"
                
        uncertainty_flag = (confidence < 0.65)
        raw_score = confidence if label == "Real" else (1.0 - confidence)

        insert_prediction_log(filename, f"Video {label}", confidence, raw_score, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        template_args = dict(
            filename=filename,
            confidence_pct=confidence_pct,
            confidence_tier=confidence_tier,
            risk_level=risk_level,
            raw_score=round(raw_score, 4),
            uncertainty_flag=uncertainty_flag,
            timestamp=now_str
        )
        if label == "Real":
            return render_template('result_real_video.html', **template_args)
        else:
            return render_template('result_fake_video.html', **template_args)
            
    except Exception as e:
        print(f"Exception during predict-video execution: {e}")
        flash(f"System error: Failed to parse uploaded video. {e}", "error")
        return redirect(url_for('predict_video_page'))

# ─── JSON API endpoints (called via AJAX fetch) ─────────────────────────────

@app.route('/api/predict-image', methods=['POST'])
def api_predict_image():
    """JSON API for image prediction — called via fetch() from frontend."""
    if not deepfake_model:
        return jsonify({"ok": False, "error": "Image model offline."}), 503
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file attached."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No file selected."}), 400
    try:
        IMG_SIZE = 160
        from PIL import Image as PILImage
        image = PILImage.open(file.stream).convert("RGB")
        image_resized = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = deepfake_model.predict(img_array, verbose=0)
        raw_score = float(prediction[0][0])
        label = "Real" if raw_score > 0.5 else "Fake"
        display_confidence = raw_score if label == "Real" else (1.0 - raw_score)
        confidence_pct = int(round(display_confidence * 100))
        if label == "Real":
            if raw_score >= 0.95: risk_level, confidence_tier = "low", "High"
            elif raw_score >= 0.70: risk_level, confidence_tier = "medium", "Medium"
            else: risk_level, confidence_tier = "high", "Low"
        else:
            if raw_score <= 0.20: risk_level, confidence_tier = "high", "High"
            elif raw_score <= 0.40: risk_level, confidence_tier = "high", "Medium"
            else: risk_level, confidence_tier = "medium", "Low"
        uncertainty_flag = (0.40 <= raw_score <= 0.65)
        filename = secure_filename(file.filename) if file.filename else "uploaded_image"
        insert_prediction_log(filename, label, display_confidence, raw_score, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)
        print(f"[API/IMAGE] {filename} → {label} ({confidence_pct}%)", flush=True)
        return jsonify({
            "ok": True, "label": label, "confidence_pct": confidence_pct,
            "confidence_tier": confidence_tier, "risk_level": risk_level,
            "raw_score": round(raw_score, 4), "uncertainty_flag": uncertainty_flag,
            "filename": filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"[API/IMAGE] Exception: {e}", flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/predict-audio', methods=['POST'])
def api_predict_audio():
    """JSON API for audio prediction — called via fetch() from frontend."""
    if not audio_model:
        return jsonify({"ok": False, "error": "Audio model offline."}), 503
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file attached."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No file selected."}), 400
    try:
        filename = secure_filename(file.filename)
        timestamp_prefix = str(int(time.time()))
        unique_filename = f"{timestamp_prefix}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)
        audio, sr = librosa.load(save_path, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(audio_device)
        with torch.inference_mode():
            logits = audio_model(audio_tensor)
            probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        label = "Real" if prediction == 0 else "Fake"
        confidence_pct = int(round(confidence * 100))
        if label == "Real":
            if confidence >= 0.95: risk_level, confidence_tier = "low", "High"
            elif confidence >= 0.70: risk_level, confidence_tier = "medium", "Medium"
            else: risk_level, confidence_tier = "high", "Low"
        else:
            if confidence >= 0.80: risk_level, confidence_tier = "high", "High"
            elif confidence >= 0.60: risk_level, confidence_tier = "high", "Medium"
            else: risk_level, confidence_tier = "medium", "Low"
        uncertainty_flag = (confidence < 0.65)
        raw_score = confidence if label == "Real" else (1.0 - confidence)
        insert_prediction_log(filename, f"Audio {label}", confidence, raw_score, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)
        print(f"[API/AUDIO] {filename} → {label} ({confidence_pct}%)", flush=True)
        return jsonify({
            "ok": True, "label": label, "confidence_pct": confidence_pct,
            "confidence_tier": confidence_tier, "risk_level": risk_level,
            "raw_score": round(raw_score, 4), "uncertainty_flag": uncertainty_flag,
            "filename": filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"[API/AUDIO] Exception: {e}", flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/predict-video', methods=['POST'])
def api_predict_video():
    """JSON API for video prediction — called via fetch() from frontend."""
    if not video_model or not video_processor:
        return jsonify({"ok": False, "error": "Video model offline."}), 503
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file attached."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"ok": False, "error": "No file selected."}), 400
    try:
        filename = secure_filename(file.filename)
        timestamp_prefix = str(int(time.time()))
        unique_filename = f"{timestamp_prefix}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)
        frames = extract_video_frames(save_path, num_frames=16)
        inputs = video_processor(list(frames), return_tensors="pt")
        inputs = {k: v.to(audio_device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = video_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        label = "Real" if prediction == 0 else "Fake"
        confidence_pct = int(round(confidence * 100))
        if label == "Real":
            if confidence >= 0.95: risk_level, confidence_tier = "low", "High"
            elif confidence >= 0.70: risk_level, confidence_tier = "medium", "Medium"
            else: risk_level, confidence_tier = "high", "Low"
        else:
            if confidence >= 0.80: risk_level, confidence_tier = "high", "High"
            elif confidence >= 0.60: risk_level, confidence_tier = "high", "Medium"
            else: risk_level, confidence_tier = "medium", "Low"
        uncertainty_flag = (confidence < 0.65)
        raw_score = confidence if label == "Real" else (1.0 - confidence)
        insert_prediction_log(filename, f"Video {label}", confidence, raw_score, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)
        print(f"[API/VIDEO] {filename} → {label} ({confidence_pct}%)", flush=True)
        return jsonify({
            "ok": True, "label": label, "confidence_pct": confidence_pct,
            "confidence_tier": confidence_tier, "risk_level": risk_level,
            "raw_score": round(raw_score, 4), "uncertainty_flag": uncertainty_flag,
            "filename": filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        print(f"[API/VIDEO] Exception: {e}", flush=True)
        return jsonify({"ok": False, "error": str(e)}), 500



@app.route('/predict-url', methods=['POST'])
def predict_url():
    if not url_model or not url_vectorizer:
        return jsonify({"error": "The URL phishing model is currently offline or not loaded on this server."}), 503
        
    url = request.form.get('url', '').strip()
    if not url:
        return jsonify({"error": "No URL address was provided."}), 400
        
    try:
        # Transform and predict
        x = url_vectorizer.transform([url])
        pred = url_model.predict(x)[0]
        
        # 1 = Safe, 0 = Malicious
        label = "Safe" if pred == 1 else "Malicious"
        
        # Calculate confidence
        confidence_score = 1.0
        if hasattr(url_model, "predict_proba"):
            proba = url_model.predict_proba(x)[0]
            confidence_score = float(proba[1] if pred == 1 else proba[0])
            
        # Log to prediction history (marked as URL source)
        log_name = f"URL: {url[:30]}..." if len(url) > 30 else f"URL: {url}"
        insert_prediction_log(log_name, label, confidence_score, pred, session.get('session_id'), session.get('user', {}).get('email') if session.get('user') else None)
        
        return jsonify({
            "url": url,
            "label": label,
            "confidence_score": round(confidence_score, 4)
        })
        
    except Exception as e:
        print(f"Exception during predict-url execution: {e}")
        return jsonify({"error": f"Internal execution failure: {str(e)}"}), 500

import uuid
from complaint_generator import generate_complaint_pdf

@app.route('/report_init', methods=['POST'])
def report_init():
    session['report_context'] = {
        'filename': request.form.get('filename'),
        'detection_type': request.form.get('detection_type'),
        'confidence': request.form.get('confidence'),
        'score': request.form.get('score'),
        'timestamp': request.form.get('timestamp')
    }
    return redirect(url_for('report'))

@app.route('/report', methods=['GET'])
def report():
    if not session.get('user'):
        session['next_url'] = url_for('report')
        flash("Please log in or register to submit an incident report.", "warning")
        return redirect(url_for('login'))
        
    context = session.get('report_context', {})
    return render_template('report.html', context=context)

@app.route('/report-media', methods=['POST'])
def report_media():
    if not session.get('user'):
        flash("Unauthorized. Please log in.", "error")
        return redirect(url_for('login'))
        
    name = request.form.get('name', '').strip()
    email = session['user']['email']  # Use authenticated email
    mobile = request.form.get('mobile', '').strip()
    address = request.form.get('address', '').strip()
    platform = request.form.get('platform', '').strip()
    description = request.form.get('description', '').strip()
    incident_type = request.form.get('type', '').strip()
    
    # Context data
    context = session.get('report_context', {})
    
    if context:
        filename = context.get('filename', '')
        detection_type = context.get('detection_type', '')
        confidence = context.get('confidence', '')
        score = context.get('score', '')
        timestamp = context.get('timestamp', '')
    else:
        detection_type = "Manual"
        confidence = ""
        score = ""
        timestamp = ""
        filename = ""
        
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename != '':
                safe_name = secure_filename(file.filename)
                timestamp_prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                unique_filename = f"report_{timestamp_prefix}_{safe_name}"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(save_path)
                filename = unique_filename
    
    try:
        complaint_id = "ISA-CMP-" + datetime.datetime.now().strftime("%Y") + "-" + str(uuid.uuid4().int)[:6]
        status = "Submitted"
        
        # Store in database
        timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_data = {
            "complaint_id": complaint_id,
            "name": name,
            "email": email,
            "mobile": mobile,
            "address": address,
            "platform": platform,
            "description": description,
            "incident_type": incident_type,
            "filename": filename,
            "detection_type": detection_type,
            "confidence": confidence,
            "score": score,
            "detection_timestamp": timestamp,
            "timestamp": timestamp_now,
            "status": status,
            "current_stage": 0,
            "timeline": [],
            "officer_stages": {"1": None, "2": None, "3": None, "4": None}
        }
        
        if mongodb_available:
            try:
                db.reports.insert_one(report_data)
            except Exception as e:
                local_incident_reports.insert(0, report_data)
        else:
            local_incident_reports.insert(0, report_data)
            
        # Generate Complaint PDF
        pdf_filename = f"Complaint_{complaint_id}.pdf"
        pdf_path = os.path.join(os.getcwd(), 'pdf_reports', pdf_filename)
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
        generate_complaint_pdf(report_data, pdf_path)
        
        # Send Email Confirmation
        from email_utils import send_confirmation_email
        email_sent = send_confirmation_email(report_data, pdf_path)
        
        # Clear context
        session.pop('report_context', None)
        
        if email_sent:
            flash(f"Incident report logged successfully! Your Complaint ID is {complaint_id}. A confirmation email has been sent.", "success")
        else:
            flash(f"Incident report logged successfully! Your Complaint ID is {complaint_id}. (Note: Email confirmation was skipped due to missing SMTP config)", "success")
            
        return redirect(url_for('history'))
        
    except Exception as e:
        print(f"Exception during report-media execution: {e}")
        flash(f"System error filing report: {e}", "error")
        return redirect(url_for('report'))

@app.route('/download_complaint/<complaint_id>')
def download_complaint(complaint_id):
    if not session.get('user'):
        return redirect(url_for('login'))
        
    # Verify user owns the complaint (basic check, could check DB)
    pdf_filename = f"Complaint_{complaint_id}.pdf"
    pdf_path = os.path.join(os.getcwd(), 'pdf_reports', pdf_filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    else:
        flash("Complaint document not found.", "error")
        return redirect(url_for('history'))

# ─── Officer Routes ──────────────────────────────────────────────────────────

STAGE_STATUS_MAP = {
    1: "Under Verification",
    2: "Under Investigation",
    3: "Action Taken",
    4: "Case Resolved",
}
STAGE_LABELS = {
    1: "Verification",
    2: "Digital Forensics Investigation",
    3: "Cybercrime Action",
    4: "Final Review & Resolution",
}

def officer_required(f):
    """Decorator: Ensures only officers can access a route."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        user = session.get('user')
        if not user or user.get('role') != 'officer':
            flash("Access denied. Officer credentials required.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_complaint_by_id(complaint_id):
    if mongodb_available:
        try:
            return db.reports.find_one({"complaint_id": complaint_id})
        except:
            pass
    return next((c for c in local_incident_reports if c.get('complaint_id') == complaint_id), None)

def update_complaint(complaint_id, update_dict):
    if mongodb_available:
        try:
            db.reports.update_one({"complaint_id": complaint_id}, {"$set": update_dict})
            return
        except:
            pass
    for c in local_incident_reports:
        if c.get('complaint_id') == complaint_id:
            c.update(update_dict)
            break

@app.route('/officer/dashboard')
@officer_required
def officer_dashboard():
    officer = session['user']
    officer_stage = officer.get('stage')
    # Fetch complaints relevant to this officer's stage
    if mongodb_available:
        try:
            import pymongo
            all_complaints = list(db.reports.find().sort("timestamp", pymongo.DESCENDING))
        except:
            all_complaints = local_incident_reports
    else:
        all_complaints = local_incident_reports

    # Pending = awaiting this officer's action (current_stage == officer_stage - 1)
    pending = [c for c in all_complaints if c.get('current_stage', 0) == officer_stage - 1]
    # Completed by this officer
    completed = [c for c in all_complaints if c.get('current_stage', 0) >= officer_stage]

    return render_template('officer_dashboard.html',
                           officer=officer,
                           pending=pending,
                           completed=completed,
                           stage_labels=STAGE_LABELS)

@app.route('/officer/complaint/<complaint_id>')
@officer_required
def officer_complaint_detail(complaint_id):
    officer = session['user']
    complaint = get_complaint_by_id(complaint_id)
    if not complaint:
        flash("Complaint not found.", "error")
        return redirect(url_for('officer_dashboard'))
    upload_folder = app.config['UPLOAD_FOLDER']
    return render_template('officer_complaint.html',
                           officer=officer,
                           complaint=complaint,
                           stage_labels=STAGE_LABELS,
                           stage_status_map=STAGE_STATUS_MAP,
                           upload_folder=upload_folder)

@app.route('/officer/complete/<complaint_id>', methods=['POST'])
@officer_required
def officer_complete_stage(complaint_id):
    officer = session['user']
    officer_stage = officer.get('stage')
    complaint = get_complaint_by_id(complaint_id)

    if not complaint:
        flash("Complaint not found.", "error")
        return redirect(url_for('officer_dashboard'))

    current_stage = complaint.get('current_stage', 0)

    # Strict server-side stage enforcement
    if current_stage != officer_stage - 1:
        flash("You are not authorized to complete this stage, or it is already completed.", "error")
        return redirect(url_for('officer_complaint_detail', complaint_id=complaint_id))

    remarks = request.form.get('remarks', '').strip()
    if not remarks:
        flash("Please provide remarks before marking the stage complete.", "warning")
        return redirect(url_for('officer_complaint_detail', complaint_id=complaint_id))

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_stage = officer_stage
    new_status = STAGE_STATUS_MAP.get(new_stage, "In Progress")

    timeline_entry = {
        "stage": officer_stage,
        "stage_label": STAGE_LABELS.get(officer_stage),
        "officer_name": officer['name'],
        "officer_title": officer.get('title', ''),
        "officer_email": officer['email'],
        "remarks": remarks,
        "completed_at": now_str,
        "status": "Completed"
    }

    # Update officer_stages sub-document
    officer_stages_update = complaint.get('officer_stages', {})
    officer_stages_update[str(officer_stage)] = timeline_entry

    # Add to timeline list
    timeline = complaint.get('timeline', [])
    timeline.append(timeline_entry)

    update_dict = {
        "current_stage": new_stage,
        "status": new_status,
        "timeline": timeline,
        "officer_stages": officer_stages_update
    }

    update_complaint(complaint_id, update_dict)

    # Refresh complaint data for email
    complaint.update(update_dict)

    # Send stage update email to the complainant
    from email_utils import send_stage_update_email
    send_stage_update_email(complaint, officer_stage, remarks)

    flash(f"Stage {officer_stage} ({STAGE_LABELS.get(officer_stage)}) marked as completed. Complainant has been notified.", "success")
    return redirect(url_for('officer_dashboard'))

# ─── Chatbot API Endpoint ───────────────────────────────────────────────────
@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided"}), 400

        user_message = data.get('message')
        history = data.get('history', [])

        if not gemini_client:
            return jsonify({"error": "Chatbot is not configured properly on the server."}), 500

        # Build context
        system_prompt = """You are the official AI Assistant for Identity Shield AI, a professional platform for detecting deepfakes (image, audio, video) and managing cybercrime complaints.
Your purpose is to help users understand the platform, how to use it, and answer questions about deepfakes and the complaint workflow.
Key facts:
- Deepfake Image Detection: Users upload an image, it uses EfficientNetB0 (TensorFlow) to predict Real or Fake. Score > 0.50 is Real.
- Deepfake Audio Detection: Uses Wav2Vec2.
- Deepfake Video Detection: Uses VideoMAE.
- Reporting/Complaints: Users can file an incident report on the 'Report' page.
- Complaint Tracking: Complainants can view status in 'History'.
- Officer Roles: Verification Officer, Digital Forensics Officer, Cybercrime Action Officer, Senior Investigating Officer.
- Be polite, professional, friendly, and concise. Do not answer questions completely unrelated to deepfakes, cybersecurity, or the platform.
"""

        # Format history for Gemini using the new SDK
        gemini_contents = []
        for msg in history:
            role = "user" if msg.get('sender') == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg.get('text')}]})
            
        gemini_contents.append({"role": "user", "parts": [{"text": user_message}]})

        # Generate response using gemini-2.5-flash
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=gemini_contents,
            config={"system_instruction": system_prompt}
        )

        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Chatbot API Error: {e}")
        return jsonify({"error": "Failed to generate response. Please try again."}), 500

# ─── Error Handler Views ─────────────────────────────────────────────────────
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(413)
def request_entity_too_large(e):
    flash("The uploaded file exceeds the maximum size limit of 50MB.", "error")
    return redirect(request.referrer or url_for('predict')), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # use_reloader=False is CRITICAL on Windows with heavy ML models.
    # The Werkzeug reloader spawns a child process that re-loads all models,
    # causing inference to potentially fail in the child process.
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
