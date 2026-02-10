
from flask import Flask, request, render_template, jsonify, url_for, redirect, flash
from flask_cors import CORS
from PIL import Image
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import os
import io
import uuid
import json
import logging
import cv2
import shutil  # ✅ ADD THIS - Required for file operations in predict route
import traceback  # ✅ ADD THIS - Required for detailed error logging
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus
from datetime import datetime
import random
from segment2 import segment_analyze_plant
# ✅ POST-HARVEST BLUEPRINT IMPORTS (ADD THESE)
from routes.post_harvest import post_harvest_bp
from routes.schemes import schemes_bp
import signal
import sys
import socket
from datetime import datetime, timedelta
from flask import session


# ✅ NEW: LOGIN & DATABASE IMPORTS
from flask_login import LoginManager, login_required, current_user, logout_user
from model import db, User, LoginHistory, DiseaseDetection, WeeklyAssessment
from routes.auth import auth_bp

from nutrition_analyzer import (
    analyze_nutrition_deficiency, 
    calculate_fertilizer_dosage, 
    load_nutrition_deficiency_data
)
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

nutrition_deficiency_data = load_nutrition_deficiency_data()
logger.info(f"Loaded {len(nutrition_deficiency_data)} nutrition deficiency types")

SERVER_START_TIME = datetime.now()

# Add this function near the top of your file, after imports
def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket to get the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
app = Flask(__name__)
CORS(app)

# ===== SESSION & DATABASE CONFIGURATION =====
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_NAME'] = 'agripal_session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
# Use environment variable for SECRET_KEY (Render provides this)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
# Get database URL from environment (for Render deployment)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///agripal.db')

# Fix for Render PostgreSQL URLs (postgres:// -> postgresql://)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SERVER_START_TIME'] = SERVER_START_TIME.isoformat()

# ===== OTHER CONFIGURATIONS =====
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
# Disable debug in production
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
app.config['DEBUG'] = (FLASK_ENV != 'production')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info(f"Upload folder configured at: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")

# ===== INITIALIZE EXTENSIONS =====
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# ===== USER LOADER =====
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Get Gemini API key from environment variable (more secure)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAL_7MfAGGI8HBpyUhAvyzUl9hPIWJk4bk")

# Configure Gemini API with error handling
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")
except Exception as e:
    logger.error(f"❌ Failed to configure Gemini API: {e}")

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess image for model input
def preprocess_image(image):
    try:
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

# Load the TensorFlow model
try:
    model_path = 'plant_diseases_model.h5'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully!")
    else:
        logger.error(f"Model file not found at: {os.path.abspath(model_path)}")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Load disease treatments from JSON file
def load_disease_treatments():
    try:
        treatment_path = 'disease_treatments.json'
        if os.path.exists(treatment_path):
            with open(treatment_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded disease treatments from {treatment_path}")
                return data
        else:
            logger.error(f"Disease treatments file not found at: {os.path.abspath(treatment_path)}")
            return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading disease treatments: {e}")
        return {}

# Class names for plant diseases
class_names = [
    "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
    "Blueberry_healthy", "Cherry_(including_sour)Powdery_mildew", "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot", "Corn(maize)_Common_rust",
    "Corn_(maize)Northern_Leaf_Blight", "Corn(maize)_healthy", "Grape_Black_rot",
    "Grape_Esca_(Black_Measles)", "Grape_Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape_healthy",
    "Orange_Haunglongbing_(Citrus_greening)", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two-spotted_spider_mite", "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus", "Tomato_healthy"
]

# Add this constant after your class_names list
CONFIDENCE_THRESHOLD = 50.0  # Minimum confidence for valid prediction
SUPPORTED_PLANTS = {
    'Apple': ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy'],
    'Blueberry': ['Blueberry_healthy'],
    'Cherry': ['Cherry_(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy'],
    'Corn (Maize)': ['Corn_(maize)Cercospora_leaf_spot_Gray_leaf_spot', 'Corn(maize)_Common_rust', 
                     'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy'],
    'Grape': ['Grape_Black_rot', 'Grape_Esca_(Black_Measles)', 'Grape_Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape_healthy'],
    'Orange': ['Orange_Haunglongbing_(Citrus_greening)'],
    'Peach': ['Peach_Bacterial_spot', 'Peach_healthy'],
    'Pepper (Bell)': ['Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy'],
    'Potato': ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy'],
    'Raspberry': ['Raspberry_healthy'],
    'Soybean': ['Soybean_healthy'],
    'Squash': ['Squash_Powdery_mildew'],
    'Strawberry': ['Strawberry_Leaf_scorch', 'Strawberry_healthy'],
    'Tomato': ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato_Target_Spot',
               'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']
}

# Common Agricultural Questions Database
COMMON_QUESTIONS = {
    "plant_diseases": [
        "What are the most common tomato diseases?",
        "How do I identify powdery mildew?",
        "What causes yellow leaves on plants?",
        "How to prevent fungal diseases in plants?",
        "What are the signs of bacterial infection in crops?",
        "How to identify viral diseases in plants?",
        "What causes leaf spots on vegetables?",
        "How to detect early blight in tomatoes?"
    ],
    "treatment_methods": [
        "What are organic pest control methods?",
        "How to make homemade fungicide?",
        "What is integrated pest management?",
        "How to use neem oil for plant diseases?",
        "What are the best copper-based fungicides?",
        "How to apply systemic pesticides safely?",
        "What is the difference between preventive and curative treatments?",
        "How to rotate pesticides to prevent resistance?"
    ],
    "crop_management": [
        "When is the best time to plant tomatoes?",
        "How much water do vegetables need daily?",
        "What is crop rotation and why is it important?",
        "How to improve soil fertility naturally?",
        "What are companion plants for tomatoes?",
        "How to prepare soil for planting?",
        "What are the signs of nutrient deficiency?",
        "How to manage weeds organically?"
    ],
    "seasonal_advice": [
        "What crops to plant in monsoon season?",
        "How to protect plants from extreme heat?",
        "What are winter crop management tips?",
        "How to prepare garden for rainy season?",
        "What vegetables grow best in summer?",
        "How to manage greenhouse in different seasons?",
        "What are post-harvest handling best practices?",
        "How to store seeds for next season?"
    ],
    "technology_agriculture": [
        "How can AI help in agriculture?",
        "What are smart farming techniques?",
        "How to use drones in agriculture?",
        "What are precision agriculture tools?",
        "How does satellite imagery help farmers?",
        "What are IoT applications in farming?",
        "How to use weather data for crop planning?",
        "What are digital farming platforms?"
    ]
}

# Load disease treatments
disease_treatments = load_disease_treatments()
logger.info(f"Loaded {len(disease_treatments)} disease treatments")

def normalize_disease_info(disease_info):
    """
    Map old JSON field names to standardized template field names
    This allows backward compatibility with your existing JSON structure
    """
    if not disease_info or 'pesticide' not in disease_info:
        return disease_info
    
    # Create deep copy to avoid modifying original
    import copy
    normalized = copy.deepcopy(disease_info)
    
    logger.info("=" * 80)
    logger.info("📄 NORMALIZING DISEASE INFO FIELDS")
    logger.info("=" * 80)
    
    # Process both chemical and organic treatments
    for treatment_type in ['chemical', 'organic']:
        if treatment_type not in normalized['pesticide']:
            logger.warning(f"⚠️ No {treatment_type} treatment found")
            continue
            
        treatment = normalized['pesticide'][treatment_type]
        logger.info(f"📦 Processing {treatment_type.upper()} treatment...")
        
        # ===== FIELD MAPPING =====
        
        # 1. Map: application_frequency -> frequency
        if 'application_frequency' in treatment and 'frequency' not in treatment:
            treatment['frequency'] = treatment['application_frequency']
            logger.info(f"  ✅ Mapped application_frequency -> frequency")
            logger.info(f"     Value: {treatment['frequency'][:50]}...")
        elif 'frequency' not in treatment or not treatment.get('frequency'):
            treatment['frequency'] = "Apply according to product label recommendations and disease pressure."
            logger.warning(f"  ⚠️ No frequency field found, added fallback")
        
        # 2. Map: precautions -> safety
        if 'precautions' in treatment and 'safety' not in treatment:
            treatment['safety'] = treatment['precautions']
            logger.info(f"  ✅ Mapped precautions -> safety")
            logger.info(f"     Value: {treatment['safety'][:50]}...")
        elif 'safety' not in treatment or not treatment.get('safety'):
            if treatment_type == 'chemical':
                treatment['safety'] = "Wear protective equipment. Follow all label precautions. Keep away from water sources."
            else:
                treatment['safety'] = "Safe for beneficial insects when used as directed. Apply during cooler parts of day."
            logger.warning(f"  ⚠️ No safety field found, added fallback")
        
        # 3. Ensure usage exists and has content
        if 'usage' not in treatment or not treatment.get('usage') or len(treatment.get('usage', '').strip()) < 10:
            treatment['usage'] = f"Apply as directed on product label. Ensure thorough coverage of all affected plant surfaces. Repeat applications as needed based on disease pressure."
            logger.warning(f"  ⚠️ Missing or short usage, added fallback")
        
        # 4. Ensure all required fields exist
        required_fields = {
            'name': f"{treatment_type.title()} Treatment",
            'dosage_per_hectare': 0.0,
            'unit': 'L',
            'usage': 'Apply as directed',
            'frequency': 'As needed',
            'safety': 'Follow product label instructions'
        }
        
        for field, default_value in required_fields.items():
            if field not in treatment or not treatment.get(field):
                treatment[field] = default_value
                logger.warning(f"  ⚠️ Missing {field}, added default: {default_value}")
        
        # Validate field lengths
        logger.info(f"  📊 Field lengths:")
        logger.info(f"     - Name: {len(treatment.get('name', ''))} chars")
        logger.info(f"     - Usage: {len(treatment.get('usage', ''))} chars")
        logger.info(f"     - Frequency: {len(treatment.get('frequency', ''))} chars")
        logger.info(f"     - Safety: {len(treatment.get('safety', ''))} chars")
    
    logger.info("=" * 80)
    logger.info("✅ NORMALIZATION COMPLETE")
    logger.info("=" * 80)
    
    return normalized
# Enhanced function to get disease information with better video source handling
def get_disease_info(disease_name):
    """
    Enhanced function with field normalization and detailed logging
    """
    try:
        logger.info("=" * 80)
        logger.info(f"🔍 DISEASE LOOKUP: {disease_name}")
        logger.info("=" * 80)
        logger.info(f"📚 Database has {len(disease_treatments)} diseases")
        
        # Try exact match first
        disease_info = disease_treatments.get(disease_name, None)
        
        # If no exact match, try variations
        if not disease_info:
            logger.info(f"⚠️ No exact match, trying variations...")
            cleaned_name = disease_name.replace('_', ' ').replace('(', '').replace(')', '').strip()
            
            for key, value in disease_treatments.items():
                if cleaned_name.lower() in key.lower() or key.lower() in cleaned_name.lower():
                    disease_info = value
                    logger.info(f"✅ Found match with key: {key}")
                    break
        
        if not disease_info:
            logger.error(f"❌ NO DISEASE INFO FOUND for: {disease_name}")
            available = list(disease_treatments.keys())[:5]
            logger.info(f"📝 Available diseases (first 5): {available}")
            return None
        
        logger.info(f"✅ Raw disease info found")
        logger.info(f"📋 Raw keys: {list(disease_info.keys())}")
        
        # ===== NORMALIZE FIELD NAMES (backward compatibility) =====
        disease_info = normalize_disease_info(disease_info)
        
        # ===== FINAL VALIDATION =====
        logger.info("=" * 80)
        logger.info("📊 FINAL VALIDATION")
        logger.info("=" * 80)
        logger.info(f"✅ Disease Name: {disease_info.get('name')}")
        logger.info(f"✅ Description: {len(disease_info.get('description', ''))} chars")
        logger.info(f"✅ Treatment Steps: {len(disease_info.get('treatment', []))}")
        logger.info(f"✅ Severity: {disease_info.get('severity')}")
        
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type in disease_info['pesticide']:
                    t = disease_info['pesticide'][treatment_type]
                    logger.info(f"")
                    logger.info(f"📦 {treatment_type.upper()}:")
                    logger.info(f"  Name: {t.get('name')}")
                    logger.info(f"  Usage: {len(t.get('usage', ''))} chars - {bool(t.get('usage'))}")
                    logger.info(f"  Frequency: {len(t.get('frequency', ''))} chars - {bool(t.get('frequency'))}")
                    logger.info(f"  Safety: {len(t.get('safety', ''))} chars - {bool(t.get('safety'))}")
                    logger.info(f"  Dosage: {t.get('dosage_per_hectare')} {t.get('unit')}/hectare")
        
        # ===== PROCESS VIDEO SOURCES =====
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type not in disease_info['pesticide']:
                    continue
                
                treatment = disease_info['pesticide'][treatment_type]
                
                if 'video_sources' in treatment:
                    video_sources = treatment['video_sources']
                    
                    # Add YouTube search URLs
                    if 'search_terms' in video_sources:
                        search_urls = []
                        for term in video_sources['search_terms']:
                            search_urls.append({
                                'term': term,
                                'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                            })
                        video_sources['search_urls'] = search_urls
                        logger.info(f"✅ Added {len(search_urls)} YouTube URLs for {treatment_type}")
                    
                    # Process reliable channels
                    if 'reliable_channels' in video_sources:
                        channel_urls = []
                        for channel in video_sources['reliable_channels']:
                            channel_urls.append({
                                'name': channel,
                                'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' ' + disease_name.replace('_', ' '))}"
                            })
                        video_sources['channel_urls'] = channel_urls
                        logger.info(f"✅ Added {len(channel_urls)} channel URLs for {treatment_type}")
        
        logger.info("=" * 80)
        logger.info("✅ DISEASE INFO PROCESSING COMPLETE")
        logger.info("=" * 80)
        
        return disease_info
        
    except Exception as e:
        logger.error(f"❌ ERROR in get_disease_info: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    
def combine_disease_treatments(unique_diseases):
    """
    Combine treatments from multiple diseases of the same plant
    Returns a merged treatment plan with intelligent deduplication
    """
    logger.info("=" * 80)
    logger.info("🔀 COMBINING TREATMENTS FROM MULTIPLE DISEASES")
    logger.info("=" * 80)
    
    combined = {
        'diseases': [],
        'description': '',
        'treatment': [],
        'severity': 'Unknown',
        'pesticide': {
            'chemical': {
                'name': 'Combined Chemical Treatment',
                'usage': [],
                'frequency': [],
                'safety': [],
                'dosage_per_hectare': 0,
                'unit': 'L',
                'video_sources': {
                    'search_terms': [],
                    'reliable_channels': []
                }
            },
            'organic': {
                'name': 'Combined Organic Treatment',
                'usage': [],
                'frequency': [],
                'safety': [],
                'dosage_per_hectare': 0,
                'unit': 'L',
                'video_sources': {
                    'search_terms': [],
                    'reliable_channels': []
                }
            }
        },
        'additional_resources': {
            'step_by_step_guide': [],
            'extension_guides': []
        }
    }
    
    severity_levels = {'Low': 1, 'Moderate': 2, 'Medium': 2, 'High': 3, 'Severe': 4}
    max_severity_score = 0
    
    # Track unique items to avoid duplicates
    unique_chemical_names = set()
    unique_organic_names = set()
    unique_treatments = set()
    unique_guides = set()
    
    logger.info(f"📊 Processing {len(unique_diseases)} diseases...")
    
    for disease, data in unique_diseases.items():
        disease_info = data['disease_info']
        if not disease_info:
            logger.warning(f"⚠️ No disease info for {disease}")
            continue
        
        logger.info(f"   Processing: {disease}")
        
        # Track diseases
        combined['diseases'].append({
            'name': disease,
            'display_name': disease.replace('_', ' '),
            'count': data['count'],
            'avg_confidence': data['total_confidence'] / data['count']
        })
        
        # Combine descriptions
        if disease_info.get('description'):
            combined['description'] += f"**{disease.replace('_', ' ')}**: {disease_info['description']}\n\n"
        
        # Combine treatment steps (with section headers)
        if disease_info.get('treatment'):
            header = f"=== Treatment for {disease.replace('_', ' ')} ==="
            if header not in unique_treatments:
                combined['treatment'].append(header)
                unique_treatments.add(header)
                
                for step in disease_info['treatment']:
                    if step and step not in unique_treatments:
                        combined['treatment'].append(step)
                        unique_treatments.add(step)
                
                combined['treatment'].append("")  # Spacer
        
        # Track highest severity
        disease_severity = disease_info.get('severity', 'Unknown')
        severity_score = severity_levels.get(disease_severity, 0)
        if severity_score > max_severity_score:
            max_severity_score = severity_score
            combined['severity'] = disease_severity
            logger.info(f"   Updated max severity: {disease_severity}")
        
        # Combine pesticide info
        if 'pesticide' in disease_info:
            for treatment_type in ['chemical', 'organic']:
                if treatment_type not in disease_info['pesticide']:
                    continue
                
                treatment = disease_info['pesticide'][treatment_type]
                unique_set = unique_chemical_names if treatment_type == 'chemical' else unique_organic_names
                
                # Collect unique treatment names and usage
                treatment_name = treatment.get('name', '')
                if treatment_name and treatment_name not in unique_set:
                    unique_set.add(treatment_name)
                    usage_text = f"**{treatment_name}** ({disease.replace('_', ' ')}): {treatment.get('usage', 'Apply as directed')}"
                    combined['pesticide'][treatment_type]['usage'].append(usage_text)
                    
                    logger.info(f"      Added {treatment_type}: {treatment_name}")
                
                # Collect frequencies
                if treatment.get('frequency'):
                    freq = treatment['frequency'].strip()
                    if freq not in combined['pesticide'][treatment_type]['frequency']:
                        combined['pesticide'][treatment_type]['frequency'].append(freq)
                
                # Collect safety info
                if treatment.get('safety'):
                    safety = treatment['safety'].strip()
                    if safety not in combined['pesticide'][treatment_type]['safety']:
                        combined['pesticide'][treatment_type]['safety'].append(safety)
                
                # Sum dosages (will be averaged later)
                dosage = treatment.get('dosage_per_hectare', 0)
                combined['pesticide'][treatment_type]['dosage_per_hectare'] += dosage
                
                # Combine video sources
                if treatment.get('video_sources'):
                    video_sources = treatment['video_sources']
                    
                    if 'search_terms' in video_sources:
                        for term in video_sources['search_terms']:
                            if term not in combined['pesticide'][treatment_type]['video_sources']['search_terms']:
                                combined['pesticide'][treatment_type]['video_sources']['search_terms'].append(term)
                    
                    if 'reliable_channels' in video_sources:
                        for channel in video_sources['reliable_channels']:
                            if channel not in combined['pesticide'][treatment_type]['video_sources']['reliable_channels']:
                                combined['pesticide'][treatment_type]['video_sources']['reliable_channels'].append(channel)
        
        # Combine additional resources
        if 'additional_resources' in disease_info:
            resources = disease_info['additional_resources']
            
            if 'step_by_step_guide' in resources:
                for step in resources['step_by_step_guide']:
                    if step not in combined['additional_resources']['step_by_step_guide']:
                        combined['additional_resources']['step_by_step_guide'].append(step)
            
            if 'extension_guides' in resources:
                for guide in resources['extension_guides']:
                    if guide not in unique_guides:
                        combined['additional_resources']['extension_guides'].append(guide)
                        unique_guides.add(guide)
    
    # Format combined fields
    logger.info("📝 Formatting combined treatment data...")
    
    for treatment_type in ['chemical', 'organic']:
        # Format usage
        if combined['pesticide'][treatment_type]['usage']:
            combined['pesticide'][treatment_type]['usage'] = "\n\n".join(
                combined['pesticide'][treatment_type]['usage']
            )
        else:
            combined['pesticide'][treatment_type]['usage'] = "Apply treatments according to product labels for each specific disease."
        
        # Format frequency
        if combined['pesticide'][treatment_type]['frequency']:
            unique_freq = list(set(combined['pesticide'][treatment_type]['frequency']))
            if len(unique_freq) == 1:
                combined['pesticide'][treatment_type]['frequency'] = unique_freq[0]
            else:
                combined['pesticide'][treatment_type]['frequency'] = " OR ".join(unique_freq)
        else:
            combined['pesticide'][treatment_type]['frequency'] = "Follow individual disease treatment schedules"
        
        # Format safety
        if combined['pesticide'][treatment_type]['safety']:
            combined['pesticide'][treatment_type]['safety'] = " • ".join(
                list(set(combined['pesticide'][treatment_type]['safety']))
            )
        else:
            combined['pesticide'][treatment_type]['safety'] = "Follow all safety guidelines on product labels. Wear protective equipment."
        
        # Average dosages
        num_diseases = len(unique_diseases)
        if num_diseases > 0 and combined['pesticide'][treatment_type]['dosage_per_hectare'] > 0:
            combined['pesticide'][treatment_type]['dosage_per_hectare'] /= num_diseases
            logger.info(f"   {treatment_type.title()} avg dosage: {combined['pesticide'][treatment_type]['dosage_per_hectare']:.2f}")
        
        # Process video sources for URLs
        video_sources = combined['pesticide'][treatment_type]['video_sources']
        if video_sources['search_terms']:
            search_urls = []
            for term in video_sources['search_terms']:
                search_urls.append({
                    'term': term,
                    'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                })
            video_sources['search_urls'] = search_urls
        
        if video_sources['reliable_channels']:
            channel_urls = []
            for channel in video_sources['reliable_channels']:
                channel_urls.append({
                    'name': channel,
                    'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' multiple plant diseases')}"
                })
            video_sources['channel_urls'] = channel_urls
    
    logger.info("=" * 80)
    logger.info("✅ COMBINED TREATMENT PLAN READY")
    logger.info(f"   Diseases: {len(combined['diseases'])}")
    logger.info(f"   Treatment steps: {len(combined['treatment'])}")
    logger.info(f"   Overall severity: {combined['severity']}")
    logger.info("=" * 80)
    
    return combined

# IMPROVED CALCULATE_DOSAGE FUNCTION
# Replace lines 651-703 in app.py with this improved version

def calculate_dosage(area, area_unit, pesticide_info):
    """Calculate pesticide dosage based on area and unit with enhanced error handling"""
    logger.info("="*60)
    logger.info("🧮 DOSAGE CALCULATION STARTED")
    logger.info("="*60)
    logger.info(f"📏 Input area: {area} {area_unit}")
    logger.info(f"📋 Pesticide info exists: {pesticide_info is not None}")
    
    try:
        chemical_dosage = None
        organic_dosage = None
        hectare_conversion = 0
        
        # Safely get pesticide information
        chemical_info = pesticide_info.get("chemical", {}) if pesticide_info else {}
        organic_info = pesticide_info.get("organic", {}) if pesticide_info else {}
        
        logger.info(f"💊 Chemical info available: {bool(chemical_info)}")
        logger.info(f"💊 Chemical info keys: {list(chemical_info.keys()) if chemical_info else []}")
        logger.info(f"🌿 Organic info available: {bool(organic_info)}")
        logger.info(f"🌿 Organic info keys: {list(organic_info.keys()) if organic_info else []}")
        
        # Get dosage per hectare with safe defaults
        chemical_dosage_per_hectare = float(chemical_info.get("dosage_per_hectare", 0))
        organic_dosage_per_hectare = float(organic_info.get("dosage_per_hectare", 0))
        
        logger.info(f"💊 Chemical dosage per hectare: {chemical_dosage_per_hectare}")
        logger.info(f"🌿 Organic dosage per hectare: {organic_dosage_per_hectare}")
        
        # Convert area to hectares with validation
        try:
            area_float = float(area) if area else 0
            if area_float <= 0:
                logger.warning(f"⚠️ Invalid or zero area value: {area}")
                logger.warning(f"⚠️ Using default 1 hectare for dosage display")
                # Use 1 hectare as default for dosage display purposes
                area_float = 1.0
                
            # Conversion factors to hectares
            conversion_factors = {
                'hectare': 1.0,
                'acre': 0.404686,
                'square_meter': 0.0001,
                'square_feet': 0.0000092903
            }
            
            hectare_conversion = area_float * conversion_factors.get(area_unit, 1.0)
            logger.info(f"📐 Converted {area_float} {area_unit} to {hectare_conversion} hectares")
            
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Error converting area to float: {e}")
            logger.error("❌ Using default 1 hectare")
            hectare_conversion = 1.0  # Default to 1 hectare
        
        # Calculate required dosage - ALWAYS calculate if dosage_per_hectare > 0
        if chemical_dosage_per_hectare > 0:
            chemical_dosage = chemical_dosage_per_hectare * hectare_conversion
            logger.info(f"✅ Calculated chemical dosage: {chemical_dosage}")
        else:
            logger.warning("⚠️ Chemical dosage_per_hectare is 0 or missing")
        
        if organic_dosage_per_hectare > 0:
            organic_dosage = organic_dosage_per_hectare * hectare_conversion
            logger.info(f"✅ Calculated organic dosage: {organic_dosage}")
        else:
            logger.warning("⚠️ Organic dosage_per_hectare is 0 or missing")
        
        logger.info("="*60)
        logger.info(f"🎯 FINAL RESULTS:")
        logger.info(f"   Chemical dosage: {chemical_dosage}")
        logger.info(f"   Organic dosage: {organic_dosage}")
        logger.info(f"   Hectare conversion: {hectare_conversion}")
        logger.info("="*60)
        
        return chemical_dosage, organic_dosage, hectare_conversion
        
    except Exception as e:
        logger.error("="*60)
        logger.error(f"❌ ERROR IN DOSAGE CALCULATION")
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        logger.error("="*60)
        return None, None, 0



# Enhanced image validation function
def is_plant_image(image_path):
    """
    Enhanced function to check if the uploaded image is likely a plant image
    using multiple validation techniques
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning("Could not read image file")
            return False
            
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. GREEN COLOR ANALYSIS (More Strict)
        # Define multiple green ranges to catch different types of plant greens
        green_ranges = [
            # Bright green (healthy leaves)
            ([35, 50, 50], [85, 255, 255]),
            # Dark green (mature leaves)
            ([25, 30, 30], [75, 255, 200]),
            # Yellow-green (some diseased leaves)
            ([15, 40, 40], [35, 255, 255])
        ]
        
        total_green_pixels = 0
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_green_pixels += cv2.countNonZero(mask)
        
        total_pixels = img.shape[0] * img.shape[1]
        green_ratio = total_green_pixels / total_pixels
        
        # 2. TEXTURE ANALYSIS - Plants have organic textures
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Calculate Local Binary Pattern variance (texture measure)
        texture_variance = np.var(gray)
        
        # 3. EDGE ANALYSIS - Natural vs artificial edges
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        edge_ratio = edge_pixels / total_pixels
        
        # 4. COLOR DISTRIBUTION ANALYSIS
        # Plants typically have more natural color distribution
        color_std = np.std(rgb, axis=(0, 1))
        color_mean = np.mean(color_std)
        
        # 5. BRIGHTNESS AND CONTRAST CHECKS
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # 6. SHAPE ANALYSIS - Look for leaf-like shapes
        # Use contour detection to find organic shapes
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        organic_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    # Organic shapes are neither too circular nor too geometric
                    if 0.1 < circularity < 0.8:
                        organic_shapes += 1
        
        # 7. GEOMETRIC PATTERN DETECTION (to reject posters/documents)
        # Look for straight lines (common in posters/documents)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        straight_lines = len(lines) if lines is not None else 0
        
        # 8. TEXT DETECTION (basic) - Posters often have text
        # Simple text detection based on horizontal edge patterns
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        text_like_pixels = cv2.countNonZero(horizontal_lines)
        text_ratio = text_like_pixels / total_pixels
        
        # DECISION LOGIC (STRICT CRITERIA)
        height, width = img.shape[:2]
        
        # Basic size check
        is_reasonable_size = height > 100 and width > 100
        
        # Green content check (STRICTER)
        has_significant_green = green_ratio > 0.12  # At least 12% green
        
        # Texture check
        has_organic_texture = texture_variance > 500  # Organic textures have variation
        
        # Edge analysis
        has_natural_edges = 0.02 < edge_ratio < 0.25  # Not too sharp, not too smooth
        
        # Color variety check
        has_natural_colors = color_mean > 15  # Natural variation in colors
        
        # Brightness check
        reasonable_brightness = 30 < brightness < 220  # Not too dark or overexposed
        
        # Contrast check
        good_contrast = contrast > 20  # Some contrast indicating detail
        
        # Organic shapes check
        has_organic_shapes = organic_shapes > 0
        
        # Reject if too many straight lines (indicates documents/posters)
        not_too_geometric = straight_lines < 10
        
        # Reject if too much text-like content
        not_text_heavy = text_ratio < 0.05  # Less than 5% text-like content
        
        # FINAL SCORING SYSTEM
        score = 0
        criteria_met = []
        
        if has_significant_green:
            score += 3
            criteria_met.append("green_content")
        
        if has_organic_texture:
            score += 2
            criteria_met.append("organic_texture")
            
        if has_natural_edges:
            score += 2
            criteria_met.append("natural_edges")
            
        if has_natural_colors:
            score += 1
            criteria_met.append("natural_colors")
            
        if reasonable_brightness:
            score += 1
            criteria_met.append("good_brightness")
            
        if good_contrast:
            score += 1
            criteria_met.append("good_contrast")
            
        if has_organic_shapes:
            score += 2
            criteria_met.append("organic_shapes")
            
        if not_too_geometric:
            score += 1
            criteria_met.append("not_geometric")
            
        if not_text_heavy:
            score += 1
            criteria_met.append("not_text_heavy")
        
        # Log detailed analysis
        logger.info(f"Plant image analysis for {image_path}:")
        logger.info(f"  - Green ratio: {green_ratio:.3f} (threshold: 0.12)")
        logger.info(f"  - Texture variance: {texture_variance:.1f} (threshold: 500)")
        logger.info(f"  - Edge ratio: {edge_ratio:.3f} (range: 0.02-0.25)")
        logger.info(f"  - Color variation: {color_mean:.1f} (threshold: 15)")
        logger.info(f"  - Brightness: {brightness:.1f} (range: 30-220)")
        logger.info(f"  - Contrast: {contrast:.1f} (threshold: 20)")
        logger.info(f"  - Organic shapes: {organic_shapes}")
        logger.info(f"  - Straight lines: {straight_lines} (threshold: <10)")
        logger.info(f"  - Text ratio: {text_ratio:.3f} (threshold: <0.05)")
        logger.info(f"  - Total score: {score}/14")
        logger.info(f"  - Criteria met: {criteria_met}")
        
        # Require minimum score of 7/14 and must have green content
        is_plant = (score >= 7 and has_significant_green and is_reasonable_size)
        
        logger.info(f"  - Final decision: {'PLANT' if is_plant else 'NOT PLANT'}")
        
        return is_plant
        
    except Exception as e:
        logger.error(f"Error in enhanced plant image validation: {e}")
        return False  # Fail safe - reject if analysis fails

# 2. ADD this new function after the is_plant_image function:

def validate_plant_type(predicted_class, confidence):
    """
    Additional validation to ensure predicted class makes sense
    """
    try:
        # Check if predicted class is in our supported classes
        if predicted_class not in class_names:
            logger.warning(f"Predicted class {predicted_class} not in supported classes")
            return False, "Predicted class not recognized"
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning(f"Confidence {confidence:.2f}% below threshold {CONFIDENCE_THRESHOLD}%")
            return False, f"Low confidence prediction ({confidence:.1f}%)"
        
        # Additional checks can be added here
        # For example, checking if the prediction pattern makes sense
        
        return True, None
        
    except Exception as e:
        logger.error(f"Error in plant type validation: {e}")
        return False, str(e)
# Enhanced preprocess function with validation
def preprocess_image_with_validation(image, image_path):
    """
    Enhanced preprocessing with strict validation
    """
    try:
        logger.info(f"Starting image validation for: {image_path}")
        
        # First check if it's likely a plant image with enhanced validation
        if not is_plant_image(image_path):
            logger.warning("Image failed plant validation - not a plant image")
            return None, False
            
        logger.info("Image passed plant validation")
        
        # Additional file format validation
        try:
            # Verify image can be opened and is valid
            image_test = Image.open(image_path)
            image_test.verify()  # This will raise an exception if image is corrupted
        except Exception as e:
            logger.error(f"Image file validation failed: {e}")
            return None, False
            
        # Original preprocessing
        image = image.resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info("Image preprocessing completed successfully")
        return img_array, True
        
    except Exception as e:
        logger.error(f"Error in enhanced image preprocessing: {e}")
        return None, False

# Enhanced prediction function
def make_enhanced_prediction(processed_image):
    """
    Enhanced prediction with multiple validation layers
    """
    try:
        logger.info("Starting enhanced prediction")
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        logger.info(f"Raw prediction - Index: {predicted_class_index}, Confidence: {confidence:.2f}%")
        
        # Get predicted class name
        if predicted_class_index >= len(class_names):
            logger.error(f"Prediction index {predicted_class_index} out of range (max: {len(class_names)-1})")
            return None, confidence, "Prediction index out of range"
        
        predicted_class = class_names[predicted_class_index]
        logger.info(f"Predicted class: {predicted_class}")
        
        # Validate the prediction
        is_valid, validation_error = validate_plant_type(predicted_class, confidence)
        if not is_valid:
            logger.warning(f"Prediction validation failed: {validation_error}")
            return None, confidence, validation_error
        
        # Additional confidence analysis - check top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_confidences = [predictions[0][i] * 100 for i in top_3_indices]
        
        logger.info("Top 3 predictions:")
        for i, (idx, conf) in enumerate(zip(top_3_indices, top_3_confidences)):
            logger.info(f"  {i+1}. {class_names[idx]}: {conf:.2f}%")
        
        # Check if there's a clear winner (gap between 1st and 2nd should be reasonable)
        if len(top_3_confidences) > 1:
            confidence_gap = top_3_confidences[0] - top_3_confidences[1]
            if confidence_gap < 10 and confidence < 70:  # If predictions are too close and confidence is low
                logger.warning(f"Ambiguous prediction - confidence gap: {confidence_gap:.2f}%")
                return None, confidence, f"Ambiguous prediction (confidence gap: {confidence_gap:.1f}%)"
        
        logger.info(f"Prediction validated successfully: {predicted_class} ({confidence:.2f}%)")
        return predicted_class, confidence, None
            
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        return None, 0.0, str(e)
def generate_gradcam(img_array, model, predicted_class_index, layer_name=None):
    """
    Generate GradCAM heatmap for disease visualization - FIXED VERSION
    
    Args:
        img_array: Preprocessed image (1, 128, 128, 3)
        model: Trained Keras model
        predicted_class_index: Index of predicted class
        layer_name: Name of last conv layer (auto-detected if None)
    
    Returns:
        heatmap: GradCAM heatmap (128, 128)
        superimposed_img: Original image with heatmap overlay
    """
    try:
        logger.info("🎯 Starting GradCAM generation...")
        
        # AUTO-DETECT last convolutional layer
        if layer_name is None:
            # Strategy 1: Look for 'conv' in name with 4D output
            conv_layers = []
            for layer in model.layers:
                try:
                    # Check if layer has output_shape and it's 4D
                    if hasattr(layer, 'output_shape'):
                        output_shape = layer.output_shape
                        # Handle both single and multiple outputs
                        if isinstance(output_shape, list):
                            output_shape = output_shape[0]
                        
                        # Check if it's a convolutional layer (4D output: batch, height, width, channels)
                        if output_shape is not None and len(output_shape) == 4:
                            if 'conv' in layer.name.lower():
                                conv_layers.append(layer.name)
                except Exception as e:
                    # Skip layers that don't have proper output_shape
                    logger.debug(f"Skipping layer {layer.name}: {e}")
                    continue
            
            # Strategy 2: If no 'conv' found, find any 4D output layer
            if not conv_layers:
                logger.info("No 'conv' layers found, searching for any 4D layer...")
                for layer in model.layers:
                    try:
                        if hasattr(layer, 'output_shape'):
                            output_shape = layer.output_shape
                            if isinstance(output_shape, list):
                                output_shape = output_shape[0]
                            
                            if output_shape is not None and len(output_shape) == 4:
                                conv_layers.append(layer.name)
                    except Exception as e:
                        continue
            
            if not conv_layers:
                logger.error("❌ No convolutional layers found in model")
                return None, None
            
            layer_name = conv_layers[-1]
            logger.info(f"✅ Auto-detected conv layer: {layer_name}")
        
        # Verify layer exists
        try:
            target_layer = model.get_layer(layer_name)
            output_shape = target_layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            logger.info(f"✅ Using layer: {layer_name} (output shape: {output_shape})")
        except Exception as e:
            logger.error(f"❌ Layer {layer_name} not found or invalid: {e}")
            return None, None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.input],
            outputs=[target_layer.output, model.output]
        )
        
        # Cast to float32 for gradient computation
        img_array = tf.cast(img_array, tf.float32)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            conv_outputs, predictions = grad_model(img_array)
            
            # Handle multiple outputs if necessary
            if isinstance(conv_outputs, list):
                conv_outputs = conv_outputs[0]
            if isinstance(predictions, list):
                predictions = predictions[0]
            
            class_channel = predictions[:, predicted_class_index]
        
        # Get gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            logger.error("❌ Gradients are None - GradCAM failed")
            return None, None
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight channels by gradient importance
        conv_outputs_value = conv_outputs[0].numpy()
        pooled_grads_value = pooled_grads.numpy()
        
        # Multiply each channel by importance and sum
        heatmap = np.zeros(conv_outputs_value.shape[:2], dtype=np.float32)
        for i in range(pooled_grads_value.shape[0]):
            heatmap += conv_outputs_value[:, :, i] * pooled_grads_value[i]
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        
        # Resize to original image size (128x128)
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        
        # Apply colormap - JET (Red = High attention, Blue = Low)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        # Get original image (denormalize from 0-1 to 0-255)
        original_img = (img_array[0].numpy() * 255).astype(np.uint8)
        original_img_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(
            original_img_bgr, 0.6,  # Original image weight
            heatmap_colored, 0.4,   # Heatmap weight
            0
        )
        
        logger.info("✅ GradCAM generated successfully!")
        logger.info(f"   - Heatmap shape: {heatmap_resized.shape}")
        logger.info(f"   - Heatmap range: [{heatmap_resized.min():.3f}, {heatmap_resized.max():.3f}]")
        
        return heatmap_resized, superimposed_img
        
    except Exception as e:
        logger.error(f"❌ Error generating GradCAM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def save_gradcam_image(superimposed_img, original_filename):
    """
    Save GradCAM visualization to uploads folder with ENHANCED ERROR HANDLING
    
    Returns:
        gradcam_filename: Filename of saved GradCAM image
    """
    try:
        # Generate filename
        base_name = os.path.splitext(original_filename)[0]
        gradcam_filename = f"{base_name}_gradcam.jpg"
        gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
        
        # Verify folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save image with high quality
        cv2.imwrite(gradcam_path, superimposed_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Verify file was saved
        if os.path.exists(gradcam_path):
            file_size = os.path.getsize(gradcam_path)
            logger.info(f"✅ GradCAM saved: {gradcam_path} ({file_size} bytes)")
            return gradcam_filename
        else:
            logger.error(f"❌ GradCAM file not found after save: {gradcam_path}")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error saving GradCAM: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_detailed_error_message(error_type, image_analysis=None):
    """
    Generate detailed error messages for different validation failures
    """
    if error_type == "not_plant":
        return {
            "title": "Not a Plant Image",
            "message": "The uploaded image doesn't appear to be a plant photograph.",
            "suggestions": [
                "Upload a clear photo of plant leaves",
                "Ensure the image shows actual plant matter (not drawings or posters)",
                "Make sure leaves are visible with any disease symptoms",
                "Use good lighting and focus on the affected plant parts"
            ],
            "technical_details": image_analysis
        }
    elif error_type == "low_confidence":
        return {
            "title": "Unable to Identify Plant Disease",
            "message": "The image quality or plant type may not be suitable for accurate analysis.",
            "suggestions": [
                "Try uploading a clearer, higher quality image",
                "Ensure the plant is one of our supported types",
                "Focus on leaves showing clear disease symptoms",
                "Check if lighting is adequate"
            ]
        }
    elif error_type == "unsupported_plant":
        return {
            "title": "Unsupported Plant Type",
            "message": "This plant type may not be in our current database.",
            "suggestions": [
                "Check our supported plants list",
                "Try with Apple, Tomato, Potato, Corn, Grape, Peach, Pepper, or Strawberry plants",
                "Ensure the image clearly shows the plant type"
            ]
        }
    else:
        return {
            "title": "Analysis Error",
            "message": "An error occurred during image analysis.",
            "suggestions": [
                "Try uploading the image again",
                "Ensure the image file is not corrupted",
                "Use a different image format (JPG, PNG)"
            ]
        }

def initialize_enhanced_gemini():
    """Enhanced Gemini AI initialization with better error handling"""
    try:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your-api-key-here":
            logger.error("Gemini API key not configured properly")
            return False, "API key not configured"
        
        genai.configure(api_key=GEMINI_API_KEY)
        
        test_model = genai.GenerativeModel('models/gemini-1.5-flash-001')
        test_prompt = "What is the most important factor in plant health?"
        
        test_response = test_model.generate_content(test_prompt)
        
        if test_response and test_response.text:
            logger.info("✅ Gemini AI connected successfully!")
            logger.info(f"Test response: {test_response.text[:100]}...")
            return True, "Connected successfully"
        else:
            logger.error("❌ Gemini AI test failed - no response received")
            return False, "No response from API"
            
    except Exception as e:
        logger.error(f"❌ Gemini AI initialization failed: {str(e)}")
        return False, str(e)

def get_enhanced_chatbot_response(message, detected_disease=None, conversation_history=None):
    """Enhanced chatbot with improved AI integration and common questions"""
    
    original_message = message
    message = message.lower().strip()
    
    logger.info(f"Enhanced chatbot processing: {original_message}")
    
    # Handle system commands first
    if message in ["help", "/help", "commands", "/commands"]:
        return generate_help_response()
    
    elif message in ["questions", "/questions", "common questions", "examples"]:
        return generate_common_questions_response()
    
    elif message.startswith("/category "):
        category = message.replace("/category ", "").strip()
        return generate_category_questions(category)
    
    # Handle date and time requests
    elif any(keyword in message for keyword in ["date", "time", "today", "current date", "current time"]):
        current_datetime = datetime.now()
        if "time" in message:
            return f"🕐 Current time: {current_datetime.strftime('%H:%M:%S')} IST"
        elif "date" in message:
            return f"📅 Today's date: {current_datetime.strftime('%B %d, %Y (%A)')}"
        else:
            return f"📅🕐 Current date and time: {current_datetime.strftime('%B %d, %Y %H:%M:%S (%A)')}"
    
    # Handle greeting responses
    elif any(greeting in message for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "namaste", "start"]):
        greeting_response = """🌱 **Namaste! Welcome to AgriPal AI!** 

I'm your intelligent agricultural assistant powered by advanced AI. I can help you with:

🔍 **Disease Detection** - Upload images for instant plant disease identification
💊 **Treatment Plans** - Get specific, science-based treatment recommendations  
🧮 **Dosage Calculator** - Calculate exact pesticide amounts for your farm size
🌿 **Organic Solutions** - Eco-friendly pest and disease management
📊 **Crop Management** - Seasonal advice and farming best practices
🤖 **AI-Powered Q&A** - Ask any agricultural question, get expert answers

**Quick Start Commands:**
- Type `questions` to see common agricultural questions
- Type `help` to see available commands
- Ask specific questions like "How to treat tomato blight?"

What would you like to explore today? 🚀"""
        return greeting_response
    
    # Handle goodbye messages
    elif any(farewell in message for farewell in ["bye", "goodbye", "see you", "thanks", "thank you", "dhanyawad"]):
        return """🙏 **Thank you for using AgriPal AI!** 

**Remember these key farming tips:**
- Monitor your crops regularly for early disease detection
- Maintain good field hygiene and crop rotation
- Keep learning about sustainable farming practices

Happy farming! 🌾🚜✨"""
    
    # For all other questions, use Gemini AI
    else:
        try:
            ai_prompt = create_agricultural_prompt(original_message, detected_disease, conversation_history)
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 500,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-001",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            response = model.generate_content(ai_prompt)
            
            if response and response.text:
                ai_response = response.text.strip()
                formatted_response = f"🤖 **AgriPal AI Expert Response:**\n\n{ai_response}\n\n---\n💡 *Need plant disease identification? Upload an image using our detection tool!*"
                return formatted_response
            else:
                return get_fallback_response(original_message, detected_disease)
                
        except Exception as e:
            logger.error(f"❌ Gemini AI error: {str(e)}")
            return get_fallback_response(original_message, detected_disease, str(e))


def get_common_questions_by_category(category=None, limit=5):
    """Get common questions by category or random selection"""
    if category and category in COMMON_QUESTIONS:
        questions = COMMON_QUESTIONS[category]
        return random.sample(questions, min(limit, len(questions)))
    else:
        all_questions = []
        for cat_questions in COMMON_QUESTIONS.values():
            all_questions.extend(cat_questions)
        return random.sample(all_questions, min(limit, len(all_questions)))

def create_agricultural_prompt(user_message, detected_disease=None, conversation_history=None):
    """Create a comprehensive prompt for agricultural AI assistant"""
    
    base_context = """You are AgriPal AI, an expert agricultural assistant specializing in:
- Plant disease identification and treatment
- Crop management and farming techniques
- Pest control and integrated pest management
- Soil health and nutrition management
- Seasonal farming advice
- Sustainable agriculture practices

Your responses should be:
- Scientifically accurate and evidence-based
- Practical and actionable for farmers
- Safe and environmentally conscious
- Appropriate for different farming scales
- Under 400 words but comprehensive"""

    disease_context = ""
    if detected_disease:
        disease_context = f"\n\nCurrent Context: The user recently detected '{detected_disease}'. This should inform your responses."

    history_context = ""
    if conversation_history:
        recent_messages = conversation_history[-3:]
        history_summary = " ".join([msg.get('text', '')[:100] for msg in recent_messages])
        history_context = f"\n\nRecent conversation: {history_summary}"

    user_context = f"\n\nUser's question: {user_message}"
    
    return base_context + disease_context + history_context + user_context

def generate_help_response():
    """Generate help response with available commands"""
    return """🆘 **AgriPal AI Help Center**

**Available Commands:**
- `help` - Show this help menu
- `questions` - View common agricultural questions
- `/category [name]` - Get questions by category

**Categories:**
- `plant_diseases` - Disease identification
- `treatment_methods` - Treatment options
- `crop_management` - Farming practices
- `seasonal_advice` - Season-specific guidance
- `technology_agriculture` - Modern farming tech

**Example Questions:**
- "What causes yellow leaves in tomatoes?"
- "How to make organic pesticide?"
- "Best time to plant vegetables?"

Just type your question naturally! 🌱"""

def generate_common_questions_response():
    """Generate response with common questions"""
    questions = get_common_questions_by_category(limit=8)
    
    response = "❓ **Popular Agricultural Questions**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"**{i}.** {question}\n"
    
    response += "\n**More Help:** Type `/category plant_diseases` for specific topics!"
    return response

def generate_category_questions(category):
    """Generate questions for a specific category"""
    if category not in COMMON_QUESTIONS:
        available_categories = ", ".join(COMMON_QUESTIONS.keys())
        return f"❓ Category '{category}' not found.\n\n**Available:** {available_categories}"
    
    questions = COMMON_QUESTIONS[category]
    category_title = category.replace('_', ' ').title()
    
    response = f"📚 **{category_title} - Questions**\n\n"
    
    for i, question in enumerate(questions, 1):
        response += f"**{i}.** {question}\n"
    
    return response

def get_fallback_response(original_message, detected_disease=None, error_msg=None):
    """Enhanced fallback response when AI is unavailable"""
    
    fallback = f"""🤖 **AgriPal AI Assistant** *(Offline Mode)*

**Your question:** "{original_message}"

"""
    
    if detected_disease:
        fallback += f"**Detected disease:** {detected_disease}\n\n"
    
    message_lower = original_message.lower()
    
    if any(word in message_lower for word in ["disease", "fungus", "infection"]):
        fallback += """**For plant diseases:**
🔍 Take clear photos of affected areas
✂️ Remove diseased plant parts
🌿 Apply appropriate treatment
📞 Consult agricultural extension officer"""
    
    elif any(word in message_lower for word in ["treatment", "pesticide", "spray"]):
        fallback += """**Treatment guidelines:**
🧪 Use registered pesticides as per label
🌱 Try organic options (neem oil, copper sulfate)
⏰ Apply during cool hours
⚠️ Always wear protective equipment"""
    
    fallback += "\n\n**Try:** `questions` for common topics or `help` for commands"
    
    return fallback

# ===== WEEKLY ASSESSMENT FUNCTIONS =====

def analyze_weekly_progress(user_id, plant_type, current_detection):
    """
    Analyze weekly progress and generate treatment recommendations
    
    Returns:
        dict with assessment, comparison, and recommendations
    """
    logger.info("=" * 80)
    logger.info("📊 WEEKLY ASSESSMENT ANALYSIS")
    logger.info("=" * 80)
    
    # Get previous weeks' assessments
    previous_assessments = WeeklyAssessment.query.filter_by(
        user_id=user_id,
        plant_type=plant_type
    ).order_by(WeeklyAssessment.week_number.desc()).limit(4).all()
    
    if not previous_assessments:
        logger.info("🆕 First assessment for this plant")
        return {
            'is_first_assessment': True,
            'week_number': 1,
            'recommendation': 'Start treatment as recommended. Take photos weekly to track progress.',
            'dosage_recommendation': 'maintain',
            'next_assessment_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        }
    
    # Get most recent assessment (last week)
    last_week = previous_assessments[0]
    week_number = last_week.week_number + 1
    
    logger.info(f"📅 Week {week_number} Assessment")
    logger.info(f"📈 Previous Week {last_week.week_number}: {last_week.severity_level}")
    
    # Calculate severity change
    severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
    
    current_severity_score = severity_map.get(current_detection['severity'], 0)
    previous_severity_score = last_week.severity_score
    
    severity_change = current_severity_score - previous_severity_score
    
    # Calculate affected area change
    current_affected = current_detection.get('color_severity', 0)
    previous_affected = last_week.color_severity_percent or 0
    area_change_percent = current_affected - previous_affected
    
    # Determine progress status
    is_improving = severity_change < 0 or area_change_percent < -5
    is_worsening = severity_change > 0 or area_change_percent > 5
    is_stable = abs(severity_change) == 0 and abs(area_change_percent) <= 5
    is_cured = current_detection.get('disease', '').lower().endswith('healthy')
    
    logger.info(f"📊 Progress Analysis:")
    logger.info(f"   - Severity change: {severity_change}")
    logger.info(f"   - Area change: {area_change_percent:+.1f}%")
    logger.info(f"   - Improving: {is_improving}")
    logger.info(f"   - Worsening: {is_worsening}")
    logger.info(f"   - Cured: {is_cured}")
    
    # Generate recommendations based on progress
    recommendation, dosage_change, switch_treatment = generate_treatment_recommendation(
        is_improving, is_worsening, is_stable, is_cured,
        week_number, last_week, current_detection
    )
    
    assessment_result = {
        'is_first_assessment': False,
        'week_number': week_number,
        'previous_week_severity': last_week.severity_level,
        'current_severity': current_detection['severity'],
        'severity_change': severity_change,
        'area_change_percent': area_change_percent,
        'is_improving': is_improving,
        'is_worsening': is_worsening,
        'is_stable': is_stable,
        'is_cured': is_cured,
        'recommendation': recommendation,
        'dosage_recommendation': dosage_change,
        'treatment_switch': switch_treatment,
        'previous_treatment': last_week.pesticide_used,
        'previous_dosage': last_week.dosage_applied,
        'next_assessment_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
        'assessment_history': [
            {
                'week': a.week_number,
                'date': a.assessment_date.strftime('%Y-%m-%d'),
                'severity': a.severity_level,
                'affected_area': a.affected_area_percent,
                'treatment': a.pesticide_used
            } for a in reversed(previous_assessments)
        ]
    }
    
    logger.info("=" * 80)
    logger.info(f"✅ Assessment Complete: {recommendation[:100]}...")
    logger.info("=" * 80)
    
    return assessment_result


def generate_treatment_recommendation(is_improving, is_worsening, is_stable, 
                                     is_cured, week_number, last_assessment, 
                                     current_detection):
    """
    Generate AI-powered treatment recommendations based on progress
    """
    
    # CURED - RECOVERY PROTOCOL
    if is_cured:
        return (
            "🎉 Excellent! Your plant has fully recovered! "
            "Continue with preventive care: maintain good hygiene, proper watering, "
            "and monitor weekly. No pesticides needed unless symptoms reappear.",
            "stop",
            None
        )
    
    # IMPROVING - CONTINUE WITH REDUCED DOSAGE
    if is_improving:
        if week_number <= 2:
            recommendation = (
                "✅ Great progress! Disease severity is decreasing. "
                "Continue with current treatment plan. "
                "Keep dosage the same for one more week to ensure effectiveness."
            )
            dosage_change = "maintain"
            switch = None
        else:
            recommendation = (
                f"✅ Continued improvement over {week_number} weeks! "
                f"You can now REDUCE pesticide dosage by 25-30% as the plant is responding well. "
                f"Previous dosage: {last_assessment.dosage_applied:.2f}L - "
                f"Reduce to: {last_assessment.dosage_applied * 0.70:.2f}L. "
                f"This reduces chemical load while maintaining effectiveness."
            )
            dosage_change = "decrease_25"
            switch = None
        
        return recommendation, dosage_change, switch
    
    # STABLE - ADJUST STRATEGY
    if is_stable:
        if week_number >= 3:
            # Not improving after 3 weeks - switch treatment
            current_type = last_assessment.pesticide_type
            switch_to = "organic" if current_type == "chemical" else "stronger chemical"
            
            recommendation = (
                f"⚠️ Disease is stable but not improving after {week_number} weeks. "
                f"Current treatment ({last_assessment.pesticide_used}) may not be fully effective. "
                f"RECOMMENDATION: Switch to {switch_to} treatment alternative. "
                f"Also increase application frequency."
            )
            dosage_change = "maintain_or_increase"
            switch = switch_to
        else:
            recommendation = (
                "📊 Disease severity is stable. Continue current treatment "
                "but monitor closely. If no improvement by next week, "
                "we'll recommend switching treatments."
            )
            dosage_change = "maintain"
            switch = None
        
        return recommendation, dosage_change, switch
    
    # WORSENING - URGENT ACTION
    if is_worsening:
        if week_number <= 2:
            recommendation = (
                "⚠️ WARNING: Disease is progressing despite treatment! "
                f"IMMEDIATE ACTION NEEDED: "
                f"1. INCREASE dosage by 30-40% "
                f"(from {last_assessment.dosage_applied:.2f}L to "
                f"{last_assessment.dosage_applied * 1.35:.2f}L). "
                f"2. Increase application frequency. "
                f"3. Remove and destroy heavily infected plant parts. "
                f"4. Improve field sanitation."
            )
            dosage_change = "increase_35"
            switch = None
        else:
            # Not responding after multiple weeks - major switch needed
            current_type = last_assessment.pesticide_type
            
            if current_type == "organic":
                recommendation = (
                    f"🚨 CRITICAL: Disease worsening after {week_number} weeks of organic treatment. "
                    f"URGENT RECOMMENDATION: Switch to CHEMICAL pesticides immediately. "
                    f"Organic methods are not controlling the infection. "
                    f"Suggested: Use systemic fungicide/pesticide for this disease. "
                    f"Consider consulting agricultural extension officer."
                )
                switch = "chemical_systemic"
            else:
                recommendation = (
                    f"🚨 CRITICAL: Disease worsening despite chemical treatment for {week_number} weeks. "
                    f"URGENT ACTIONS: "
                    f"1. Switch to DIFFERENT chemical class (avoid resistance) "
                    f"2. Increase dosage by 40% "
                    f"3. Apply every 5 days instead of weekly "
                    f"4. Consider professional consultation "
                    f"5. Test if disease strain is pesticide-resistant"
                )
                switch = "different_chemical_class"
            
            dosage_change = "increase_40"
        
        return recommendation, dosage_change, switch
    
    # DEFAULT FALLBACK
    return (
        "Continue monitoring. Take clear photos weekly for accurate tracking.",
        "maintain",
        None
    )


def save_weekly_assessment(user_id, plant_type, detection_data, assessment_result):
    """Save the weekly assessment to database"""
    try:
        severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
        
        assessment = WeeklyAssessment(
            user_id=user_id,
            plant_type=plant_type,
            disease_name=detection_data.get('disease', 'Unknown'),
            week_number=assessment_result['week_number'],
            assessment_date=datetime.now(),
            
            # Severity
            severity_level=detection_data.get('severity', 'Unknown'),
            severity_score=severity_map.get(detection_data.get('severity', 'Unknown'), 0),
            color_severity_percent=detection_data.get('color_severity', 0),
            affected_area_percent=detection_data.get('affected_percentage', 0),
            
            # Treatment (from form data)
            pesticide_used=detection_data.get('pesticide_used', 'Not specified'),
            pesticide_type=detection_data.get('pesticide_type', 'chemical'),
            dosage_applied=detection_data.get('dosage_applied', 0),
            application_method=detection_data.get('application_method', 'Spray'),
            
            # Progress
            is_improving=assessment_result.get('is_improving', False),
            is_worsening=assessment_result.get('is_worsening', False),
            is_stable=assessment_result.get('is_stable', False),
            is_cured=assessment_result.get('is_cured', False),
            
            # Recommendations
            recommendation=assessment_result.get('recommendation', ''),
            recommended_dosage_change=assessment_result.get('dosage_recommendation', 'maintain'),
            recommended_switch=assessment_result.get('treatment_switch'),
            
            # Images
            image_filename=detection_data.get('image_filename'),
            
            # Notes
            farmer_notes=detection_data.get('farmer_notes', '')
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        logger.info(f"✅ Weekly assessment saved: Week {assessment_result['week_number']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving weekly assessment: {e}")
        db.session.rollback()
        return False

def startup_gemini_check():
    """Check Gemini AI status on startup"""
    logger.info("🚀 Initializing Enhanced AgriPal Chatbot...")
    
    success, message = initialize_enhanced_gemini()
    
    if success:
        logger.info(f"✅ Chatbot Ready: {message}")
    else:
        logger.warning(f"⚠️ AI Limited Mode: {message}")
    
    return success

# Enhanced function to get disease information with better debugging
def get_disease_info(disease_name):
    """Enhanced function to get disease information with proper JSON structure handling"""
    try:
        logger.info(f"Looking for disease info: {disease_name}")
        logger.info(f"Available diseases in JSON: {list(disease_treatments.keys())[:5]}...")  # Log first 5 keys
        
        # First, try exact match
        disease_info = disease_treatments.get(disease_name, None)
        
        # If no exact match, try with some variations
        if not disease_info:
            logger.info(f"No exact match for {disease_name}, trying variations...")
            # Try removing underscores and parentheses for matching
            cleaned_name = disease_name.replace('_', ' ').replace('(', '').replace(')', '').strip()
            for key, value in disease_treatments.items():
                if cleaned_name.lower() in key.lower() or key.lower() in cleaned_name.lower():
                    disease_info = value
                    logger.info(f"Found match with key: {key}")
                    break
        
        if disease_info:
            # Validate the structure
            required_keys = ['name', 'description', 'treatment', 'severity', 'pesticide']
            missing_keys = [key for key in required_keys if key not in disease_info]
            if missing_keys:
                logger.warning(f"Missing keys in disease info: {missing_keys}")
            
            # Process video sources for better YouTube integration
            if 'pesticide' in disease_info:
                for treatment_type in ['chemical', 'organic']:
                    if treatment_type in disease_info['pesticide']:
                        treatment = disease_info['pesticide'][treatment_type]
                        
                        # Process video sources if they exist
                        if 'video_sources' in treatment:
                            video_sources = treatment['video_sources']
                            
                            # Add YouTube search URLs for each search term
                            if 'search_terms' in video_sources:
                                search_urls = []
                                for term in video_sources['search_terms']:
                                    search_urls.append({
                                        'term': term,
                                        'url': f"https://www.youtube.com/results?search_query={quote_plus(term)}"
                                    })
                                video_sources['search_urls'] = search_urls
                                logger.info(f"Added {len(search_urls)} search URLs for {treatment_type}")
                            
                            # Process reliable channels for direct links
                            if 'reliable_channels' in video_sources:
                                channel_urls = []
                                for channel in video_sources['reliable_channels']:
                                    channel_urls.append({
                                        'name': channel,
                                        'url': f"https://www.youtube.com/results?search_query={quote_plus(channel + ' ' + disease_name.replace('_', ' '))}"
                                    })
                                video_sources['channel_urls'] = channel_urls
                                logger.info(f"Added {len(channel_urls)} channel URLs for {treatment_type}")
            
            logger.info(f"Successfully processed disease info for: {disease_name}")
            return disease_info
        
        logger.warning(f"No disease info found for: {disease_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error processing disease info for {disease_name}: {e}")
        return None

# ===== REGISTER BLUEPRINTS =====
app.register_blueprint(auth_bp)
app.register_blueprint(post_harvest_bp)
app.register_blueprint(schemes_bp)

# ===== SESSION MANAGEMENT =====
def clear_sessions_on_startup():
    session_dir = './.flask_session/'
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            logger.info("🗑️ Cleared all previous sessions")
        except Exception as e:
            logger.error(f"❌ Error clearing sessions: {e}")
    os.makedirs(session_dir, exist_ok=True)

def init_database():
    with app.app_context():
        db.create_all()
        logger.info("✅ Database tables created successfully")

# ===== SESSION VALIDATION MIDDLEWARE =====
@app.before_request
def validate_session():
    if request.endpoint and (
        request.endpoint.startswith('static') or 
        request.endpoint.startswith('auth.') or
        request.endpoint == 'index' or
        request.endpoint == 'health_check' or
        request.endpoint == 'api_info'
    ):
        return
    
    if current_user.is_authenticated:
        if 'session_start' not in session:
            logout_user()
            session.clear()
            flash('Your session has expired. Please login again.', 'warning')
            return redirect(url_for('auth.login'))
        
        session_server_start = session.get('server_start')
        current_server_start = app.config['SERVER_START_TIME']
        
        if session_server_start != current_server_start:
            logout_user()
            session.clear()
            flash('Server was restarted. Please login again.', 'info')
            return redirect(url_for('auth.login'))
        
# ===== DISEASE HISTORY COMPARISON =====
def check_previous_detection(user_id, plant_type):
    """Check if user has previous detection of same plant within last 30 days"""
    try:
        one_month_ago = datetime.now() - timedelta(days=30)
        
        previous = DiseaseDetection.query.filter(
            DiseaseDetection.user_id == user_id,
            DiseaseDetection.plant_type == plant_type,
            DiseaseDetection.detection_time >= one_month_ago
        ).order_by(DiseaseDetection.detection_time.desc()).first()
        
        if previous:
            days_ago = (datetime.now() - previous.detection_time).days
            logger.info(f"📊 Found previous {plant_type} detection from {days_ago} days ago")
            return True, previous, days_ago
        
        return False, None, 0
        
    except Exception as e:
        logger.error(f"Error checking previous detection: {e}")
        return False, None, 0

def compare_disease_progress(previous_detection, current_severity, current_disease):
    """Compare current detection with previous and generate feedback"""
    severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Severe': 4}
    
    prev_severity_score = severity_map.get(previous_detection.severity, 0)
    curr_severity_score = severity_map.get(current_severity, 0)
    
    comparison = {
        'previous_disease': previous_detection.detected_disease,
        'previous_severity': previous_detection.severity,
        'current_disease': current_disease,
        'current_severity': current_severity,
        'days_since_last': (datetime.now() - previous_detection.detection_time).days,
        'improved': False,
        'worsened': False,
        'same': False,
        'message': '',
        'recommendation': ''
    }
    
    # Check if same disease
    if previous_detection.detected_disease == current_disease:
        if curr_severity_score < prev_severity_score:
            comparison['improved'] = True
            comparison['message'] = f"🎉 Great news! Your {previous_detection.plant_type} is improving! Severity reduced from {previous_detection.severity} to {current_severity}."
            comparison['recommendation'] = "Continue with your current treatment plan. Keep monitoring regularly."
        
        elif curr_severity_score > prev_severity_score:
            comparison['worsened'] = True
            comparison['message'] = f"⚠️ Alert: Disease severity has increased from {previous_detection.severity} to {current_severity}."
            comparison['recommendation'] = "Current treatment may not be effective. Consider switching to stronger alternatives or consult an expert."
        
        else:
            comparison['same'] = True
            comparison['message'] = f"📊 Disease severity remains {current_severity}."
            comparison['recommendation'] = "Continue treatment. If no improvement in next week, consider alternative methods."
    
    else:
        # Different disease detected
        if 'healthy' in current_disease.lower():
            comparison['improved'] = True
            comparison['message'] = f"🌟 Excellent! Your plant has recovered from {previous_detection.detected_disease}!"
            comparison['recommendation'] = "Maintain good crop management practices to prevent future infections."
        else:
            comparison['worsened'] = True
            comparison['message'] = f"⚠️ New disease detected: {current_disease} (previously: {previous_detection.detected_disease})"
            comparison['recommendation'] = "Multiple diseases detected. Implement comprehensive disease management strategy."
    
    return comparison

# ===== HEALTH CHECK ENDPOINT FOR RENDER =====
@app.route('/health')
def health_check():
    """Health check endpoint for Render monitoring"""
    try:
        # Check database connection
        db_status = 'connected'
        try:
            db.session.execute('SELECT 1')
        except:
            db_status = 'disconnected'
        
        # Check model status
        model_status = 'loaded' if model is not None else 'not_loaded'
        
        # Calculate uptime
        uptime = datetime.now() - SERVER_START_TIME
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': int(uptime.total_seconds()),
            'database': db_status,
            'model': model_status,
            'gemini_api': 'configured' if GEMINI_API_KEY else 'not_configured'
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503
# All your existing routes
@app.route('/')
def index():
    """Landing page - shows beautiful page for guests, dashboard for logged-in users"""
    if current_user.is_authenticated:
        # If user is logged in, redirect to dashboard
        logger.info(f"Authenticated user {current_user.username} accessing root - redirecting to dashboard")
        return redirect(url_for('dashboard'))
    
    # Guest users see the landing page
    logger.info("Guest user accessing landing page")
    return render_template('index2.html')
# Enhanced chatbot routes
@app.route('/chatbot')
@login_required  # ✅ ADD THIS
def chatbot_page():
    logger.info(f"User {current_user.username} accessing chatbot")
    return render_template('chatbot.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with statistics and history"""
    total_detections = DiseaseDetection.query.filter_by(user_id=current_user.id).count()
    recent_detections = DiseaseDetection.query.filter_by(user_id=current_user.id)\
        .order_by(DiseaseDetection.detection_time.desc()).limit(10).all()
    
    # Weekly assessments count
    total_assessments = WeeklyAssessment.query.filter_by(user_id=current_user.id).count()
    
    # Disease statistics
    disease_stats = db.session.query(
        DiseaseDetection.detected_disease,
        db.func.count(DiseaseDetection.id).label('count')
    ).filter_by(user_id=current_user.id)\
     .group_by(DiseaseDetection.detected_disease)\
     .order_by(db.text('count DESC'))\
     .limit(5).all()
    
    # Plant type statistics
    plant_stats = db.session.query(
        DiseaseDetection.plant_type,
        db.func.count(DiseaseDetection.id).label('count')
    ).filter_by(user_id=current_user.id)\
     .group_by(DiseaseDetection.plant_type)\
     .all()
    
    # Weekly assessments grouped by plant type
    weekly_assessments_raw = WeeklyAssessment.query.filter_by(user_id=current_user.id)\
        .order_by(WeeklyAssessment.plant_type, WeeklyAssessment.week_number.desc())\
        .all()
    
    # Group assessments by plant type
    weekly_assessments = {}
    for assessment in weekly_assessments_raw:
        plant_type = assessment.plant_type
        if plant_type not in weekly_assessments:
            weekly_assessments[plant_type] = []
        weekly_assessments[plant_type].append(assessment)
    
    return render_template('dashboard.html',
        total_detections=total_detections,
        total_assessments=total_assessments,
        recent_detections=recent_detections,
        disease_stats=disease_stats,
        plant_stats=plant_stats,
        weekly_assessments=weekly_assessments,
        timedelta=timedelta  # Pass timedelta for template calculations
    )

@app.route('/api/chat/enhanced', methods=['POST'])
def enhanced_chat_api():
    """Enhanced API endpoint with better features"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        conversation_history = data.get('history', [])
        detected_disease = data.get('detected_disease')
        
        # Try to get detected disease from file if not provided
        if not detected_disease:
            try:
                with open('detected_disease.json', 'r') as f:
                    disease_data = json.load(f)
                    detected_disease = disease_data.get('disease')
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        response_text = get_enhanced_chatbot_response(
            user_message, 
            detected_disease, 
            conversation_history
        )
        
        return jsonify({
            'success': True,
            'response': response_text,
            'timestamp': datetime.now().isoformat(),
            'detected_disease': detected_disease,
            'ai_status': 'online' if gemini_status else 'offline'
        })
        
    except Exception as e:
        logger.error(f"Enhanced Chat API error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing your message',
            'details': str(e)
        }), 500

@app.route('/api/chat/common-questions')
def get_common_questions_api():
    """API to get common questions by category"""
    try:
        category = request.args.get('category')
        limit = int(request.args.get('limit', 10))
        
        if category:
            if category not in COMMON_QUESTIONS:
                return jsonify({
                    'success': False,
                    'error': f'Category not found: {category}',
                    'available_categories': list(COMMON_QUESTIONS.keys())
                }), 404
            
            questions = get_common_questions_by_category(category, limit)
            return jsonify({
                'success': True,
                'category': category,
                'questions': questions,
                'total': len(questions)
            })
        else:
            all_categories = {}
            for cat, questions in COMMON_QUESTIONS.items():
                all_categories[cat] = {
                    'title': cat.replace('_', ' ').title(),
                    'sample_questions': questions[:3],
                    'total_questions': len(questions)
                }
            
            return jsonify({
                'success': True,
                'categories': all_categories,
                'total_questions': sum(len(q) for q in COMMON_QUESTIONS.values())
            })
            
    except Exception as e:
        logger.error(f"Common questions API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
    
@app.route('/api/chat/system-status')
def chat_status():
    """Get current chat status and context"""
    try:
        detected_disease = None
        detection_time = None
        
        try:
            with open('detected_disease.json', 'r') as f:
                disease_data = json.load(f)
                detected_disease = disease_data.get('disease')
                detection_time = disease_data.get('timestamp')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        return jsonify({
            'success': True,
            'gemini_available': GEMINI_API_KEY is not None,
            'detected_disease': detected_disease,
            'detection_time': detection_time,
            'model_loaded': model is not None,
            'supported_plants': len(SUPPORTED_PLANTS)
        })
    except Exception as e:
        logger.error(f"Chat status error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/api/user-data', methods=['GET'])
@login_required
def get_user_data():
    """Fetch user's saved location and land size."""
    try:
        user = current_user
        
        user_data = {
            'location': user.location if hasattr(user, 'location') and user.location else '',
            'land_area': user.land_area if hasattr(user, 'land_area') and user.land_area else 0,
            'area_unit': user.area_unit if hasattr(user, 'area_unit') and user.area_unit else 'square_meter'
        }
        
        return jsonify(user_data), 200
        
    except Exception as e:
        return jsonify({
            'location': '',
            'land_area': 0,
            'area_unit': 'square_meter'
        }), 200

@app.route('/detection-tool')
@login_required  # ✅ ADD THIS
def detection_tool():
    logger.info(f"User {current_user.username} accessing detection tool")
    return render_template('detection-tool.html')

@app.route('/detection')
def detection():
    logger.info("Rendering detection page")
    return render_template('detection-tool.html')

@app.route('/about-us')
def about_us():
    logger.info("Rendering about-us page")
    return render_template('about-us.html')

@app.route('/contact')
def contact():
    logger.info("Rendering contact page")
    return render_template('contact.html')

@app.route('/library')
def library():
    logger.info("Rendering library page")
    return render_template('library.html')

# ✅ POST-HARVEST MANAGEMENT ROUTES (ADD THESE)

@app.route('/post-harvest')
def post_harvest_page():
    """Render post-harvest management page"""
    logger.info("Rendering post-harvest management page")
    return render_template('post-harvest.html')

@app.route('/schemes')
def schemes_page():
    """Render government schemes page"""
    logger.info("Rendering schemes page")
    return render_template('schemes.html')

@app.route('/api/info')
def api_info():
    """Enhanced API information endpoint"""
    return jsonify({
        'message': 'AGRI_PAL Unified API',
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            'disease_detection': {
                'predict': 'POST /predict',
                'supported_plants': 'GET /api/supported-plants',
                'treatment': 'GET /api/treatment/<disease_name>'
            },
            'post_harvest': {
                'agro_shops': 'POST /post-harvest/agro-shops',
                'markets': 'POST /post-harvest/markets',
                'storage': 'POST /post-harvest/storage'
            },
            'schemes': {
                'all_schemes': 'GET /api/schemes',
                'categories': 'GET /api/schemes/categories',
                'by_category': 'GET /api/schemes/category/<category>',
                'by_id': 'GET /api/schemes/<scheme_id>',
                'search': 'GET /api/schemes/search?q=<query>'
            },
            'chatbot': {
                'chat': 'POST /api/chat/enhanced',
                'common_questions': 'GET /api/chat/common-questions',
                'status': 'GET /api/chat/system-status'
            }
        }
    })

@app.route('/plant-library')
def plant_library():
    logger.info("Rendering plant library page")
    return render_template('library.html')

@app.route('/process_audio', methods=['POST'])


# Line 1260
@app.route('/api/supported-plants')
def get_supported_plants():
    """API endpoint to get list of supported plants"""
    return jsonify({
        'supported_plants': SUPPORTED_PLANTS,
        'total_plants': len(SUPPORTED_PLANTS),
        'total_conditions': len(class_names)
    })
# Line 1267 ends here
# Line 1268 (NEW)
@app.route('/upload')
def upload_file():
    """
    Route for upload page - alias for detection tool
    This fixes the BuildError in error.html
    """
    logger.info("Upload file route accessed - redirecting to detection tool")
    return detection_tool()
# Line 1275 (NEW) ends here
from flask_login import login_required, current_user
@app.route('/predict', methods=['POST'])
@login_required
def analyze():
    """Enhanced predict endpoint with FastSAM segmentation, multi-disease detection, weekly assessment tracking, and user data saving"""
    
    # Clear any previous detection at the start
    try:
        if os.path.exists('detected_disease.json'):
            os.remove('detected_disease.json')
    except Exception as e:
        logger.error(f"Error removing old detection file: {e}")
    
    logger.info("=" * 80)
    logger.info("🚀 ENHANCED PREDICT ENDPOINT - MULTI-DISEASE DETECTION + WEEKLY ASSESSMENT + USER DATA SAVE")
    logger.info("=" * 80)
    
    if model is None:
        flash("Error: Model not loaded.", "error")
        return render_template("error.html", back_link="/detection-tool")
    
    if 'image' not in request.files:
        flash("Error: No image file uploaded.", "error")
        return render_template("error.html", back_link="/detection-tool")
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        flash("Error: No image selected.", "error")
        return render_template("error.html", back_link="/detection-tool")
    
    if not allowed_file(image_file.filename):
        flash("Error: Invalid file type.", "error")
        return render_template("error.html", back_link="/detection-tool")
    
    try:
        # ===== STEP 0: GET AND SAVE USER DATA (LOCATION & LAND SIZE) =====
        logger.info("💾 Extracting and saving user data...")
        
        # Get form data
        location = request.form.get("location", "").strip()
        area = request.form.get("area", "0")
        area_unit = request.form.get("area_unit", "square_meter")
        area_float = float(area) if area else 0.0
        
        # Save to user profile for future auto-fill
        try:
            user_data_updated = False
            
            # Save location if provided
            if location and location != '':
                if not hasattr(current_user, 'location') or current_user.location != location:
                    current_user.location = location
                    user_data_updated = True
                    logger.info(f"📍 Updated user location: {location}")
            
            # Save land area if provided and valid
            if area_float > 0:
                if not hasattr(current_user, 'land_area') or current_user.land_area != area_float:
                    current_user.land_area = area_float
                    user_data_updated = True
                    logger.info(f"📏 Updated user land area: {area_float}")
            
            # Save area unit if provided
            if area_unit:
                if not hasattr(current_user, 'area_unit') or current_user.area_unit != area_unit:
                    current_user.area_unit = area_unit
                    user_data_updated = True
                    logger.info(f"📐 Updated user area unit: {area_unit}")
            
            # Commit changes to database
            if user_data_updated:
                db.session.commit()
                logger.info("✅ User profile data saved successfully")
            else:
                logger.info("ℹ️ User profile data unchanged")
                
        except Exception as user_save_error:
            logger.warning(f"⚠️ Could not save user profile data: {user_save_error}")
            db.session.rollback()
            # Continue anyway - this is not critical
        
        # ===== STEP 1: SAVE IMAGE =====
        image_filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        logger.info(f"✅ Image saved to: {image_path}")
        
        # ===== STEP 2: SEGMENTATION =====
        logger.info("🔬 Starting segmentation with severity analysis...")
        leaf_results, plant_severity, plant_level = segment_analyze_plant(image_path)
        
        leaves_dir = os.path.join("static", "individual_leaves")
        leaf_paths = [r["leaf"] for r in leaf_results]
        
        logger.info(f"✅ Segmentation completed - {len(leaf_results)} leaves found")
        logger.info(f"🌱 Plant Severity: {plant_severity}% ({plant_level})")
        
        predictions = []
        leaf_severities = []
        plant_types_detected = {}
        
        # ===== STEP 3: LEAF-WISE PREDICTION =====
        logger.info("🔍 Starting disease detection on each leaf...")
        
        for idx, leaf_path in enumerate(leaf_paths, 1):
            logger.info(f"   Processing leaf {idx}/{len(leaf_paths)}: {os.path.basename(leaf_path)}")
            
            try:
                leaf_image = Image.open(leaf_path).convert("RGB")
                processed_image, is_valid = preprocess_image_with_validation(leaf_image, leaf_path)
                
                if not is_valid:
                    logger.warning(f"   ⚠️ Leaf {idx} failed plant validation - skipping")
                    continue
                
                predicted_class, confidence, error_message = make_enhanced_prediction(processed_image)
                
                if error_message:
                    logger.warning(f"   ⚠️ Leaf {idx}: Prediction failed - {error_message}")
                    continue
                
                plant_type = predicted_class.split("_")[0]
                disease_info = get_disease_info(predicted_class)
                severity = disease_info.get("severity", "Unknown") if disease_info else "Unknown"
                
                # Get segmentation data for this leaf
                seg_result = leaf_results[idx - 1] if idx - 1 < len(leaf_results) else None
                color_severity = seg_result.get("severity_percent", 0) if seg_result else 0
                affected_percentage = seg_result.get("affected_percentage", 0) if seg_result else 0
                color_severity_level = seg_result.get("severity_level", "Unknown") if seg_result else "Unknown"
                leaf_area = seg_result.get("leaf_area", 0) if seg_result else 0
                
                prediction_data = {
                    "leaf": os.path.basename(leaf_path),
                    "leaf_number": idx,
                    "predicted_class": predicted_class,
                    "plant_type": plant_type,
                    "confidence": confidence,
                    "model_severity": severity,
                    "disease_info": disease_info,
                    "color_severity": color_severity,
                    "affected_percentage": affected_percentage,
                    "color_severity_level": color_severity_level,
                    "leaf_area": leaf_area,
                    "has_segmentation": seg_result is not None
                }
                
                predictions.append(prediction_data)
                plant_types_detected.setdefault(plant_type, []).append(prediction_data)
                
                logger.info(f"   ✅ Leaf {idx}: {predicted_class} ({confidence:.1f}%) - "
                           f"Plant: {plant_type}, Model: {severity}, Color: {color_severity_level}")
                
                if severity != "Unknown":
                    leaf_severities.append(severity)
                    
            except Exception as leaf_error:
                logger.error(f"   ❌ Error processing leaf {idx}: {leaf_error}")
                continue
        
        logger.info(f"📊 Disease detection completed: {len(predictions)}/{len(leaf_paths)} leaves analyzed")
        
        if not predictions:
            logger.error("❌ No valid predictions from any leaf")
            flash("No valid predictions could be made. Please try another image.", "error")
            return render_template("error.html", back_link="/detection-tool")
        
        # ===== STEP 4: DOMINANT PLANT TYPE =====
        if len(plant_types_detected) > 1:
            logger.info("🔍 Multiple plant types detected - filtering to dominant type...")
            dominant_plant_type = max(
                plant_types_detected.items(),
                key=lambda x: sum(p["confidence"] for p in x[1])
            )[0]
            logger.info(f"🎯 Dominant plant type: {dominant_plant_type}")
            predictions = [p for p in predictions if p["plant_type"] == dominant_plant_type]
        else:
            dominant_plant_type = list(plant_types_detected.keys())[0]
        
        # ===== STEP 5: UNIQUE DISEASES =====
        unique_diseases = {}
        for p in predictions:
            d = p["predicted_class"]
            if d not in unique_diseases:
                unique_diseases[d] = {
                    "count": 0,
                    "total_confidence": 0,
                    "disease_info": p["disease_info"],
                    "severities": [],
                    "leaves": []
                }
            unique_diseases[d]["count"] += 1
            unique_diseases[d]["total_confidence"] += p["confidence"]
            unique_diseases[d]["leaves"].append(p["leaf_number"])
            if p["model_severity"] != "Unknown":
                unique_diseases[d]["severities"].append(p["model_severity"])
        
        predicted_class = max(
            unique_diseases.items(),
            key=lambda x: (x[1]["count"], x[1]["total_confidence"])
        )[0]
        
        confidence = max(predictions, key=lambda x: x["confidence"])["confidence"]
        
        logger.info(f"🦠 Unique diseases detected: {len(unique_diseases)}")
        logger.info(f"🎯 Primary disease: {predicted_class} ({confidence:.1f}%)")
        
        # ===== STEP 6: COMBINE TREATMENTS IF MULTIPLE DISEASES =====
        combined_treatments = None
        if len(unique_diseases) > 1:
            logger.info("🔀 Multiple diseases detected - combining treatments...")
            combined_treatments = combine_disease_treatments(unique_diseases)
        
        # ===== STEP 7: OVERALL SEVERITY =====
        severity_map = {"Low": 1, "Moderate": 2, "High": 3, "Severe": 4}
        if leaf_severities:
            avg = sum(severity_map.get(s, 0) for s in leaf_severities) / len(leaf_severities)
            overall_severity = (
                "Low" if avg <= 1.5 else
                "Moderate" if avg <= 2.5 else
                "High" if avg <= 3.5 else
                "Severe"
            )
        else:
            overall_severity = "Unknown"
        
        logger.info(f"📈 Overall Severity: {overall_severity}")
        
        # ===== STEP 8: PREVIOUS DETECTION + COMPARISON =====
        has_previous, previous_detection, days_ago = check_previous_detection(
            current_user.id, dominant_plant_type
        )
        
        comparison_data = None
        if has_previous and days_ago >= 7:
            comparison_data = compare_disease_progress(
                previous_detection, overall_severity, predicted_class
            )
        
        # ===== STEP 9: GRADCAM =====
        gradcam_filename = None
        full_image = Image.open(image_path).convert("RGB")
        processed_full, _ = preprocess_image_with_validation(full_image, image_path)
        
        if processed_full is not None:
            try:
                class_index = class_names.index(predicted_class)
                _, overlay = generate_gradcam(processed_full, model, class_index)
                if overlay is not None:
                    gradcam_filename = save_gradcam_image(overlay, image_filename)
                    if gradcam_filename:
                        logger.info(f"✅ GradCAM saved: {gradcam_filename}")
            except Exception as gradcam_error:
                logger.error(f"❌ GradCAM error: {gradcam_error}")
        
        # ===== STEP 10: DOSAGE =====
        disease_info = get_disease_info(predicted_class)
        chemical_dosage = organic_dosage = hectare_conversion = None
        
        if disease_info and area_float > 0:
            chemical_dosage, organic_dosage, hectare_conversion = calculate_dosage(
                area_float, area_unit, disease_info["pesticide"]
            )
        
        # ===== STEP 11: SAVE FOR CHATBOT =====
        with open("detected_disease.json", "w") as f:
            json.dump({
                "disease": predicted_class,
                "confidence": confidence,
                "severity": overall_severity,
                "plant_type": dominant_plant_type,
                "timestamp": str(datetime.now()),
                "total_leaves_analyzed": len(predictions),
                "multiple_diseases": len(unique_diseases) > 1,
                "unique_disease_count": len(unique_diseases)
            }, f, indent=2)
        
        # ===== STEP 12: SAVE TO DATABASE =====
        detection = DiseaseDetection(
            user_id=current_user.id,
            detected_disease=predicted_class,
            confidence=confidence,
            severity=overall_severity,
            plant_type=dominant_plant_type,
            image_filename=image_filename,
            gradcam_filename=gradcam_filename,
            farm_area=area_float,
            farm_area_unit=area_unit,
            farm_location=location,
            total_leaves_analyzed=len(predictions),
            unique_diseases_count=len(unique_diseases),
            is_multi_disease=len(unique_diseases) > 1,
            chemical_dosage=chemical_dosage,
            organic_dosage=organic_dosage
        )
        
        db.session.add(detection)
        db.session.commit()
        
        logger.info("✅ Disease detection saved to database")
        
        # ===== STEP 13: WEEKLY ASSESSMENT TRACKING =====
        logger.info("=" * 80)
        logger.info("📊 RUNNING WEEKLY ASSESSMENT ANALYSIS")
        logger.info("=" * 80)
        
        assessment_result = analyze_weekly_progress(
            current_user.id,
            dominant_plant_type,
            {
                'disease': predicted_class,
                'severity': overall_severity,
                'color_severity': plant_severity,
                'affected_percentage': plant_severity,
                'image_filename': image_filename
            }
        )
        
        logger.info(f"📋 Assessment Result: Week {assessment_result['week_number']}")
        logger.info(f"💡 Recommendation: {assessment_result['recommendation'][:100]}...")
        
        # Save the weekly assessment to database
        assessment_saved = save_weekly_assessment(
            current_user.id,
            dominant_plant_type,
            {
                'disease': predicted_class,
                'severity': overall_severity,
                'color_severity': plant_severity,
                'affected_percentage': plant_severity,
                'image_filename': image_filename,
                'pesticide_used': request.form.get('previous_pesticide', 'Not specified'),
                'pesticide_type': request.form.get('pesticide_type', 'chemical'),
                'dosage_applied': float(request.form.get('dosage_used', 0) or 0),
                'application_method': request.form.get('application_method', 'Spray'),
                'farmer_notes': request.form.get('notes', '')
            },
            assessment_result
        )
        
        if assessment_saved:
            logger.info("✅ Weekly assessment saved to database")
        else:
            logger.warning("⚠️ Weekly assessment could not be saved")
        
        logger.info("=" * 80)
        
        # ===== STEP 14: PREPARE HEATMAP URLS =====
        segmented_dir = os.path.join("static", "segmented_output")
        heatmap_path = os.path.join(segmented_dir, "segmented_leaf_heatmap.png")
        has_heatmap = os.path.exists(heatmap_path)
        
        # ===== STEP 15: RENDER RESULT =====
        logger.info("=" * 80)
        logger.info("📦 RENDERING RESULTS")
        logger.info(f"✅ Disease: {predicted_class}")
        logger.info(f"✅ Confidence: {confidence:.1f}%")
        logger.info(f"✅ Severity: {overall_severity}")
        logger.info(f"✅ Leaves: {len(predictions)}")
        logger.info(f"✅ Plant: {dominant_plant_type}")
        logger.info(f"✅ Location: {location}")
        logger.info(f"✅ Land Area: {area_float} {area_unit}")
        logger.info(f"✅ Assessment Week: {assessment_result['week_number']}")
        logger.info("=" * 80)
        
        return render_template(
            "result1.html",
            # Images
            image_url=url_for("static", filename=f"uploads/{image_filename}"),
            gradcam_url=url_for("static", filename=f"uploads/{gradcam_filename}") if gradcam_filename else None,
            heatmap_url=url_for("static", filename="segmented_output/segmented_leaf_heatmap.png") if has_heatmap else None,
            segmented_image_url=url_for("static", filename="segmented_output/segmented_leaf.png") if os.path.exists(os.path.join("static", "segmented_output", "segmented_leaf.png")) else None,
            
            # Prediction data
            predicted_class=predicted_class,
            confidence=confidence,
            severity=overall_severity,
            total_leaves=len(predictions),
            all_predictions=predictions,
            
            # Plant & disease tracking
            dominant_plant_type=dominant_plant_type,
            unique_diseases=unique_diseases if len(unique_diseases) > 1 else None,
            combined_treatments=combined_treatments,
            is_multi_disease=len(unique_diseases) > 1,
            
            # Plant severity (color-based)
            plant_severity=plant_severity,
            plant_severity_level=plant_level,
            
            # Disease information
            result=disease_info,
            
            # Farm data
            area=area,
            area_unit=area_unit,
            location=location,
            
            # Dosage calculations
            chemical_dosage=chemical_dosage,
            organic_dosage=organic_dosage,
            hectare_conversion=hectare_conversion,
            
            # Leaf results
            leaf_results=leaf_results,
            
            # Comparison data
            has_previous=has_previous,
            comparison=comparison_data,
            days_since_last=days_ago,
            
            # ===== WEEKLY ASSESSMENT DATA =====
            assessment=assessment_result,
            has_assessment=True,
            show_dosage_change=assessment_result.get('dosage_recommendation') != 'maintain'
        )
    
    except Exception as e:
        db.session.rollback()
        logger.error("=" * 80)
        logger.error("❌ ERROR IN PREDICT ROUTE")
        logger.error("=" * 80)
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        flash("Unexpected error occurred during analysis.", "error")
        return render_template("error.html", back_link="/detection-tool"), 500
    
@app.route('/health')
def health_check():
    health_status = {
        "status": "ok",
        "model_loaded": model is not None,
        "treatments_loaded": len(disease_treatments) > 0,
        "upload_dir_exists": os.path.exists(app.config['UPLOAD_FOLDER']),
        "total_diseases": len(disease_treatments)
    }
    return jsonify(health_status)


@app.route('/api/treatment/<disease_name>')
def treatment_api(disease_name):
    try:
        logger.info(f"Treatment API called for disease: {disease_name}")
        disease_info = get_disease_info(disease_name)
        if disease_info:
            return jsonify(disease_info)
        else:
            logger.warning(f"No disease info found for: {disease_name}")
            return jsonify({'error': 'Disease information not found'}), 404
    except Exception as e:
        logger.error(f"Treatment API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/resources/<disease_name>')
def resources_api(disease_name):
    try:
        logger.info(f"Resources API called for disease: {disease_name}")
        disease_info = get_disease_info(disease_name)
        if disease_info and 'additional_resources' in disease_info:
            return jsonify(disease_info['additional_resources'])
        else:
            return jsonify({'error': 'Additional resources not found'}), 404
    except Exception as e:
        logger.error(f"Resources API error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/chat/direct-ai', methods=['POST'])
def direct_ai_chat():
    """Direct Gemini AI chat without AgriPal formatting"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not message:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
            
        # Simple direct prompt to Gemini
        model = genai.GenerativeModel('gemini-1.5-flash-001')
        
        # Build conversation context
        conversation_context = ""
        if history:
            recent_messages = history[-3:]  # Last 3 exchanges
            for msg in recent_messages:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['text']}\n"
        
        # Create direct prompt
        full_prompt = f"""You are Gemini AI, a helpful and knowledgeable AI assistant. Respond naturally and comprehensively to the user's question.
        {conversation_context}
        Human: {message}
        Assistant:"""
        
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            return jsonify({
                'success': True,
                'response': response.text.strip(),
                'timestamp': datetime.now().isoformat(),
                'ai_status': 'online',
                'mode': 'direct_ai'
            })
        else:
            return jsonify({'success': False, 'error': 'No response from AI'}), 500
            
    except Exception as e:
        logger.error(f"Direct AI chat error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# New endpoint for dosage calculation
@app.route('/api/calculate-dosage', methods=['POST'])
def calculate_dosage_api():
    try:
        data = request.json
        disease_name = data.get('disease_name')
        area = data.get('area')
        area_unit = data.get('area_unit', 'hectare')
        
        disease_info = get_disease_info(disease_name)
        if disease_info and 'pesticide' in disease_info:
            chemical_dosage, organic_dosage, hectare_conversion = calculate_dosage(
                area, area_unit, disease_info['pesticide']
            )
            return jsonify({
                'chemical_dosage': chemical_dosage,
                'organic_dosage': organic_dosage,
                'hectare_conversion': hectare_conversion,
                'area': area,
                'area_unit': area_unit
            })
        else:
            return jsonify({'error': 'Disease or pesticide information not found'}), 404
    except Exception as e:
        logger.error(f"Dosage calculation API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ROUTE 1: Nutrition Testing Page
# ============================================================================
@app.route('/nutrition-testing')
@login_required
def nutrition_testing():
    """Nutrition deficiency testing tool page"""
    return render_template('nutrition_testing.html', 
                         user=current_user,
                         location=current_user.location if hasattr(current_user, 'location') else '',
                         land_area=current_user.land_area if hasattr(current_user, 'land_area') else 0,
                         area_unit=current_user.area_unit if hasattr(current_user, 'area_unit') else 'square_meter')


# ============================================================================
# ROUTE 2: Analyze Nutrition Deficiency
# ============================================================================
@app.route('/analyze-nutrition', methods=['POST'])
@login_required
def analyze_nutrition():
    """Analyze uploaded image for nutrition deficiency"""
    
    logger.info("=" * 80)
    logger.info("🔬 NUTRITION DEFICIENCY ANALYSIS ENDPOINT")
    logger.info("=" * 80)
    
    if 'image' not in request.files:
        flash("Error: No image file uploaded.", "error")
        return render_template("error.html", back_link="/nutrition-testing")
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        flash("Error: No image selected.", "error")
        return render_template("error.html", back_link="/nutrition-testing")
    
    if not allowed_file(image_file.filename):
        flash("Error: Invalid file type. Please upload PNG, JPG, or JPEG.", "error")
        return render_template("error.html", back_link="/nutrition-testing")
    
    try:
        # Get form data
        location = request.form.get("location", "").strip()
        area = request.form.get("area", "0")
        area_unit = request.form.get("area_unit", "square_meter")
        area_float = float(area) if area else 0.0
        
        # Save user data
        try:
            if location:
                current_user.location = location
            if area_float > 0:
                current_user.land_area = area_float
                current_user.area_unit = area_unit
            db.session.commit()
            logger.info("✅ User profile data saved")
        except Exception as e:
            logger.warning(f"⚠️ Could not save user data: {e}")
            db.session.rollback()
        
        # Save uploaded image
        image_filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image_file.save(image_path)
        logger.info(f"✅ Image saved to: {image_path}")
        
        # Analyze nutrition deficiency
        analysis_result = analyze_nutrition_deficiency(image_path)
        
        if not analysis_result['success']:
            flash(f"Error during analysis: {analysis_result.get('error', 'Unknown error')}", "error")
            return render_template("error.html", back_link="/nutrition-testing")
        
        diagnoses = analysis_result['diagnoses']
        
        if len(diagnoses) == 0:
            # No deficiency detected - plant appears healthy
            return render_template('nutrition_result.html',
                                 healthy=True,
                                 image_url=url_for('static', filename=f'uploads/{image_filename}'),
                                 location=location,
                                 area=area,
                                 area_unit=area_unit,
                                 color_analysis=analysis_result['color_analysis'])
        
        # Calculate fertilizer dosages for primary deficiency
        primary_deficiency = diagnoses[0]
        
        if area_float > 0:
            chemical_dosage, organic_dosage, hectare_conversion = calculate_fertilizer_dosage(
                area_float, 
                area_unit, 
                primary_deficiency['fertilizer']
            )
        else:
            chemical_dosage = None
            organic_dosage = None
            hectare_conversion = 0
        
        # Render results
        return render_template('nutrition_result.html',
                             healthy=False,
                             diagnoses=diagnoses,
                             primary_deficiency=primary_deficiency,
                             image_url=url_for('static', filename=f'uploads/{image_filename}'),
                             location=location,
                             area=area,
                             area_unit=area_unit,
                             chemical_dosage=chemical_dosage,
                             organic_dosage=organic_dosage,
                             hectare_conversion=hectare_conversion,
                             color_analysis=analysis_result['color_analysis'])
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ ERROR IN NUTRITION ANALYSIS")
        logger.error("=" * 80)
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        flash("Unexpected error occurred during nutrition analysis.", "error")
        return render_template("error.html", back_link="/nutrition-testing"), 500


# ============================================================================
# ROUTE 3: API Endpoint for Nutrition Info
# ============================================================================
@app.route('/api/nutrition/<deficiency_key>')
def nutrition_api(deficiency_key):
    """API endpoint to get nutrition deficiency information"""
    try:
        logger.info(f"Nutrition API called for: {deficiency_key}")
        
        if deficiency_key in nutrition_deficiency_data:
            return jsonify(nutrition_deficiency_data[deficiency_key])
        else:
            logger.warning(f"No nutrition info found for: {deficiency_key}")
            return jsonify({'error': 'Nutrition deficiency information not found'}), 404
            
    except Exception as e:
        logger.error(f"Nutrition API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTE 4: Calculate Fertilizer Dosage API
# ============================================================================
@app.route('/api/calculate-fertilizer', methods=['POST'])
def calculate_fertilizer_api():
    """API endpoint to calculate fertilizer dosage"""
    try:
        data = request.json
        deficiency_key = data.get('deficiency_key')
        area = data.get('area')
        area_unit = data.get('area_unit', 'hectare')
        
        if deficiency_key in nutrition_deficiency_data:
            deficiency_info = nutrition_deficiency_data[deficiency_key]
            
            chemical_dosage, organic_dosage, hectare_conversion = calculate_fertilizer_dosage(
                float(area), area_unit, deficiency_info['fertilizer']
            )
            
            return jsonify({
                'success': True,
                'chemical_dosage': chemical_dosage,
                'organic_dosage': organic_dosage,
                'hectare_conversion': hectare_conversion,
                'area': area,
                'area_unit': area_unit
            })
        else:
            return jsonify({'success': False, 'error': 'Deficiency information not found'}), 404
            
    except Exception as e:
        logger.error(f"Fertilizer calculation API error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Add missing import for datetime
from datetime import datetime

# Initialize enhanced system
gemini_status = startup_gemini_check()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n" + "="*80)
    logger.info("🛑 SHUTDOWN SIGNAL RECEIVED (Ctrl+C)")
    logger.info("="*80)
    logger.info("Cleaning up resources...")

    try:
        temp_files = ['detected_disease.json']
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"✅ Cleaned up: {temp_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    logger.info("👋 AgriPal Server Stopped Successfully!")
    logger.info("="*80)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    
    # Get port from environment variable (REQUIRED for Render)
    port = int(os.environ.get('PORT', 5000))
    
    # Clear sessions and initialize database
    clear_sessions_on_startup()
    init_database()   # Creates database tables
    
    # Get local IP for display
    local_ip = get_local_ip()
    
    # Environment detection
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    IS_PRODUCTION = (FLASK_ENV == 'production')

    logger.info("="*80)
    logger.info("🌱 STARTING AGRIPAL APPLICATION 🌱")
    logger.info("="*80)
    logger.info(f"Environment: {FLASK_ENV}")
    logger.info(f"Port: {port}")
    logger.info(f"Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    logger.info(f"Disease treatments loaded: {len(disease_treatments)}")
    logger.info(f"Gemini AI status: {'✅ Online' if gemini_status else '⚠️ Offline'}")
    logger.info("="*80)
    
    if not IS_PRODUCTION:
        # Show local/network URLs only in development
        logger.info("📱 ACCESS URLs:")
        logger.info(f"   Local:    http://127.0.0.1:{port}")
        logger.info(f"   Network:  http://{local_ip}:{port}")
        logger.info("="*80)
        logger.info("📱 MOBILE ACCESS:")
        logger.info(f"   1. Connect your phone to the SAME WiFi network")
        logger.info(f"   2. Open browser on phone")
        logger.info(f"   3. Go to: http://{local_ip}:{port}")
        logger.info("="*80)
        logger.info("🛑 SHUTDOWN: Press Ctrl+C to stop the server")
        logger.info("="*80)

    try:
        app.run(
            host='0.0.0.0',  # CRITICAL: Must be 0.0.0.0 for Render
            port=port,       # CRITICAL: Must use PORT from environment
            debug=not IS_PRODUCTION,  # Disable debug in production
            use_reloader=not IS_PRODUCTION  # Disable reloader in production
        )
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)
