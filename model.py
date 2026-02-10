"""
Database models for user authentication and tracking
File: models.py
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User account model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    location = db.Column(db.String(200))
    farm_size = db.Column(db.Float)
    farm_size_unit = db.Column(db.String(20), default='hectare')
    
    # Account metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    login_count = db.Column(db.Integer, default=0)
    
    # Relationships
    login_history = db.relationship('LoginHistory', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    disease_detections = db.relationship('DiseaseDetection', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def update_login(self):
        """Update login timestamp and count"""
        self.last_login = datetime.utcnow()
        self.login_count += 1
        db.session.commit()
    
    def __repr__(self):
        return f'<User {self.username}>'


class LoginHistory(db.Model):
    """Track user login history"""
    __tablename__ = 'login_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.String(256))
    location = db.Column(db.String(200))
    
    def __repr__(self):
        return f'<LoginHistory {self.user_id} at {self.login_time}>'


class DiseaseDetection(db.Model):
    """Store disease detection history"""
    __tablename__ = 'disease_detections'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Detection details
    detected_disease = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(50))
    plant_type = db.Column(db.String(100))
    
    # Image info
    image_filename = db.Column(db.String(256))
    gradcam_filename = db.Column(db.String(256))
    
    # Farm details
    farm_area = db.Column(db.Float)
    farm_area_unit = db.Column(db.String(20))
    farm_location = db.Column(db.String(200))
    
    # Analysis results
    total_leaves_analyzed = db.Column(db.Integer)
    unique_diseases_count = db.Column(db.Integer)
    is_multi_disease = db.Column(db.Boolean, default=False)
    
    # Treatment applied
    treatment_type = db.Column(db.String(50))  # 'chemical', 'organic', or 'both'
    chemical_dosage = db.Column(db.Float)
    organic_dosage = db.Column(db.Float)
    
    # Metadata
    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)

# Add this class to your model.py file
class WeeklyAssessment(db.Model):
    """Track weekly plant assessments for treatment progress monitoring"""
    __tablename__ = 'weekly_assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    plant_type = db.Column(db.String(100), nullable=False)
    disease_name = db.Column(db.String(200), nullable=False)
    
    # Week tracking
    week_number = db.Column(db.Integer, nullable=False)
    assessment_date = db.Column(db.DateTime, default=datetime.now)
    
    # Severity tracking
    severity_level = db.Column(db.String(50))
    severity_score = db.Column(db.Integer)
    color_severity_percent = db.Column(db.Float)
    affected_area_percent = db.Column(db.Float)
    
    # Treatment tracking
    pesticide_used = db.Column(db.String(200))
    pesticide_type = db.Column(db.String(50))
    dosage_applied = db.Column(db.Float)
    application_method = db.Column(db.String(200))
    
    # Progress indicators
    is_improving = db.Column(db.Boolean, default=False)
    is_worsening = db.Column(db.Boolean, default=False)
    is_stable = db.Column(db.Boolean, default=False)
    is_cured = db.Column(db.Boolean, default=False)
    
    # Recommendations
    recommendation = db.Column(db.Text)
    recommended_dosage_change = db.Column(db.String(100))
    recommended_switch = db.Column(db.String(200))
    
    # Images
    image_filename = db.Column(db.String(255))
    comparison_image = db.Column(db.String(255))
    
    # Notes
    farmer_notes = db.Column(db.Text)
    
    # Relationships
    user = db.relationship('User', backref='weekly_assessments')
    
    def __repr__(self):
        return f'<Detection {self.detected_disease} by User {self.user_id}>'