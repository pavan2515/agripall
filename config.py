"""
AgriPal RAG System - Configuration File
PRODUCTION VERSION for Render Deployment
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "agricultural_docs"
CHUNKS_DIR = BASE_DIR / "chunks" / "processed"
VECTOR_DB_PATH = str(BASE_DIR / "chroma_agri_db")

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

FLASK_ENV = os.getenv('FLASK_ENV', 'development')
IS_PRODUCTION = FLASK_ENV == 'production'

# ============================================================================
# DATABASE CONFIGURATION (Updated for Render)
# ============================================================================

# Get database URL from environment variable
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///agripal.db')

# Fix for Render PostgreSQL URLs (postgres:// -> postgresql://)
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# MySQL Configuration (for local development only)
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', 'pavan'),
    'database': os.getenv('MYSQL_DATABASE', 'agripal_rag'),
    'port': int(os.getenv('MYSQL_PORT', 3306))
}

# ============================================================================
# API KEYS (Use Environment Variables)
# ============================================================================

# Gemini API Key - REQUIRED
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    if IS_PRODUCTION:
        raise ValueError("❌ GEMINI_API_KEY environment variable must be set in production!")
    else:
        print("⚠️  Warning: GEMINI_API_KEY not set. Some features may not work.")

# ✅ UPDATED: Use correct Gemini 2.5 model names
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

# Alternative models (in order of preference)
GEMINI_FALLBACK_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
    "models/gemini-flash-latest",
    "models/gemini-pro-latest",
    "models/gemini-2.0-flash",
]

# Claude API Key (Optional)
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ============================================================================
# SECRET KEY (for Flask sessions)
# ============================================================================

SECRET_KEY = os.getenv('SECRET_KEY')

if not SECRET_KEY:
    if IS_PRODUCTION:
        raise ValueError("❌ SECRET_KEY environment variable must be set in production!")
    else:
        SECRET_KEY = os.urandom(24)
        print("⚠️  Using randomly generated SECRET_KEY for development")

# ============================================================================
# RAG SYSTEM CONFIGURATION
# ============================================================================

# Text Chunking Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_SIZE = 200

# Embedding Model (for vector search)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# RAG RETRIEVAL SETTINGS
# ============================================================================

TOP_K_DOCUMENTS = 5
SIMILARITY_THRESHOLD = 0.3
MAX_CONTEXT_TOKENS = 8000

# Gemini API Settings
GEMINI_TEMPERATURE = 0.7
GEMINI_TOP_P = 0.9
GEMINI_TOP_K = 40
GEMINI_MAX_OUTPUT_TOKENS = 800

# ============================================================================
# AGRICULTURAL DOMAIN SETTINGS
# ============================================================================

SUPPORTED_CROPS = [
    'Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane',
    'Tomato', 'Potato', 'Onion', 'Chilli', 'Brinjal',
    'Mango', 'Banana', 'Coconut', 'Tea', 'Coffee',
    'Groundnut', 'Soybean', 'Sunflower', 'Mustard',
    'Chickpea', 'Lentil', 'Pigeon Pea', 'Green Gram'
]

INDIAN_STATES = [
    'Karnataka', 'Maharashtra', 'Tamil Nadu', 'Kerala', 'Gujarat',
    'Rajasthan', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar',
    'West Bengal', 'Madhya Pradesh', 'Andhra Pradesh', 'Telangana',
    'Odisha', 'Chhattisgarh', 'Jharkhand', 'Assam', 'Himachal Pradesh'
]

SEASONS = ['Kharif', 'Rabi', 'Zaid', 'Summer', 'Winter', 'Monsoon']

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Enable/disable verbose logging
VERBOSE_LOGGING = not IS_PRODUCTION

# ============================================================================
# CHATBOT SETTINGS
# ============================================================================

MAX_CONVERSATION_HISTORY = 10
ENABLE_CHAT_HISTORY_DB = True
ENABLE_FALLBACK_RESPONSES = True

# Intent detection keywords
INTENT_KEYWORDS = {
    'scheme': ['scheme', 'subsidy', 'yojana', 'benefit', 'government', 'pm-kisan', 'pmfby'],
    'market_price': ['price', 'mandi', 'market', 'rate', 'sell', 'buy', 'cost'],
    'disease': ['disease', 'pest', 'infection', 'treatment', 'cure', 'control', 'fungus'],
    'crop_guide': ['cultivation', 'farming', 'planting', 'sowing', 'harvest', 'grow'],
    'fertilizer': ['fertilizer', 'nutrient', 'soil', 'compost', 'npk', 'organic'],
    'weather': ['weather', 'rainfall', 'temperature', 'climate', 'forecast'],
}

# ============================================================================
# FILE UPLOAD CONFIGURATION (for Render)
# ============================================================================

UPLOAD_FOLDER = os.path.join('static', 'uploads')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================================
# PRODUCTION SETTINGS
# ============================================================================

if IS_PRODUCTION:
    # Security settings
    SESSION_COOKIE_SECURE = True  # HTTPS only
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Performance settings
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year cache for static files
    
    # Disable debug
    DEBUG = False
else:
    DEBUG = True
    SESSION_COOKIE_SECURE = False

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    warnings = []
    
    print("\n" + "="*70)
    print("🔍 CONFIGURATION VALIDATION")
    print("="*70)
    
    print(f"\n🌍 Environment: {FLASK_ENV}")
    print(f"🏠 Base Directory: {BASE_DIR}")
    
    # Check Gemini API key
    if GEMINI_API_KEY:
        print(f"✅ Gemini API Key: Configured")
        print(f"🤖 Primary Model: {GEMINI_MODEL}")
    else:
        errors.append("❌ Gemini API key not configured!")
        print(f"❌ Gemini API Key: NOT CONFIGURED")
    
    # Check database
    print(f"\n🗄️  Database Configuration:")
    if DATABASE_URL.startswith('sqlite'):
        print(f"   Type: SQLite")
        warnings.append("⚠️  Using SQLite - data may be lost on Render redeploys")
    elif DATABASE_URL.startswith('postgresql'):
        print(f"   Type: PostgreSQL")
        print(f"   ✅ Production-ready database configured")
    
    # Check directories
    print(f"\n📁 Directories:")
    print(f"   Data: {DATA_DIR} {'✅' if DATA_DIR.exists() else '⚠️  (will be created)'}")
    print(f"   Uploads: {UPLOAD_FOLDER} {'✅' if os.path.exists(UPLOAD_FOLDER) else '⚠️  (will be created)'}")
    
    # Security check
    if IS_PRODUCTION:
        print(f"\n🔐 Production Security:")
        print(f"   HTTPS Cookies: {'✅' if SESSION_COOKIE_SECURE else '❌'}")
        print(f"   Debug Mode: {'❌ ENABLED (DANGER!)' if DEBUG else '✅ Disabled'}")
    
    # Print warnings
    if warnings:
        print(f"\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    # Print errors
    if errors:
        print(f"\n❌ Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Fix these errors before deploying!")
        print("="*70)
        return False
    
    print("\n✅ Configuration validated successfully!")
    print("="*70)
    return True


def print_config_summary():
    """Print a summary of current configuration"""
    print("\n" + "="*70)
    print("🌾 AgriPal RAG System - Configuration Summary")
    print("="*70)
    
    print(f"\n🌍 Environment: {FLASK_ENV}")
    print(f"🐛 Debug Mode: {DEBUG}")
    
    print(f"\n📂 Paths:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Data: {DATA_DIR}")
    print(f"   Uploads: {UPLOAD_FOLDER}")
    
    print(f"\n🤖 AI Configuration:")
    print(f"   Gemini Model: {GEMINI_MODEL}")
    print(f"   Temperature: {GEMINI_TEMPERATURE}")
    print(f"   Max Tokens: {GEMINI_MAX_OUTPUT_TOKENS}")
    
    print(f"\n📚 RAG Configuration:")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Top-K Retrieval: {TOP_K_DOCUMENTS}")
    
    print(f"\n🗄️  Database:")
    if DATABASE_URL.startswith('sqlite'):
        print(f"   Type: SQLite (Development)")
    elif DATABASE_URL.startswith('postgresql'):
        print(f"   Type: PostgreSQL (Production)")
    
    print(f"\n🌾 Domain Knowledge:")
    print(f"   Crops: {len(SUPPORTED_CROPS)}")
    print(f"   States: {len(INDIAN_STATES)}")
    print(f"   Seasons: {len(SEASONS)}")
    
    print("="*70)


if __name__ == "__main__":
    print_config_summary()
    print()
    validate_config()
