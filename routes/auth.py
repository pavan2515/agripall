"""
Authentication routes and user management
File: routes/auth.py
COMPLETE VERSION with 10-digit phone validation
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, current_app
from flask_login import login_user, logout_user, login_required, current_user
from model import db, User, LoginHistory, DiseaseDetection
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


def validate_phone_number(phone):
    """
    Validate phone number - must be exactly 10 digits
    Returns: (is_valid, cleaned_phone, error_message)
    """
    if not phone or phone.strip() == '':
        return True, None, None  # Optional field
    
    # Remove all non-digit characters
    cleaned = re.sub(r'\D', '', phone)
    
    # Check if exactly 10 digits
    if len(cleaned) != 10:
        return False, None, f"Phone number must be exactly 10 digits (got {len(cleaned)})"
    
    # Check if all digits are valid
    if not cleaned.isdigit():
        return False, None, "Phone number must contain only digits"
    
    return True, cleaned, None


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login page with session tracking"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = request.form.get('remember', False)
        
        if not username or not password:
            flash('Please enter both username and password', 'error')
            return render_template('auth/login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact support.', 'error')
                return render_template('auth/login.html')
            
            # ✅ Log in user
            login_user(user, remember=remember)
            
            # ✅ SET SESSION DATA - Track session and server instance
            session['session_start'] = datetime.now().isoformat()
            session['server_start'] = current_app.config.get('SERVER_START_TIME')
            session['user_id'] = user.id
            session['username'] = user.username
            
            # Update login info
            user.update_login()
            
            # Record login history
            login_record = LoginHistory(
                user_id=user.id,
                ip_address=request.remote_addr,
                user_agent=request.user_agent.string[:256],
                location=request.form.get('location', '')
            )
            db.session.add(login_record)
            db.session.commit()
            
            logger.info(f"✅ User logged in: {username}")
            logger.info(f"   Session Start: {session['session_start']}")
            logger.info(f"   Server Start: {session['server_start']}")
            
            flash(f'Welcome back, {user.full_name or username}!', 'success')
            
            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard'))
        
        flash('Invalid username or password', 'error')
        logger.warning(f"⚠️ Failed login attempt: {username}")
    
    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Logout user and clear session"""
    username = current_user.username
    
    # ✅ Clear all session data
    session.clear()
    
    # Logout user
    logout_user()
    
    logger.info(f"👋 User logged out: {username}")
    logger.info(f"   Session cleared completely")
    
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page with 10-digit phone validation"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        location = request.form.get('location', '').strip()
        farm_size = request.form.get('farm_size', '')
        farm_size_unit = request.form.get('farm_size_unit', 'hectare')
        
        # Validation
        errors = []
        
        # Username validation
        if not username or len(username) < 3:
            errors.append('Username must be at least 3 characters')
        
        # Email validation
        if not email or '@' not in email:
            errors.append('Valid email is required')
        
        # Password validation
        if not password or len(password) < 6:
            errors.append('Password must be at least 6 characters')
        
        if password != confirm_password:
            errors.append('Passwords do not match')
        
        # Check for existing username
        if User.query.filter_by(username=username).first():
            errors.append('Username already exists')
        
        # Check for existing email
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered')
        
        # ✅ PHONE NUMBER VALIDATION (10 digits)
        phone_valid, cleaned_phone, phone_error = validate_phone_number(phone)
        if not phone_valid:
            errors.append(phone_error)
        
        # Display all errors
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/register.html')
        
        try:
            # Create new user
            new_user = User(
                username=username,
                email=email,
                full_name=full_name,
                phone=cleaned_phone,  # ✅ Store cleaned 10-digit phone
                location=location,
                farm_size=float(farm_size) if farm_size else None,
                farm_size_unit=farm_size_unit
            )
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            logger.info(f"✅ New user registered: {username}")
            if cleaned_phone:
                logger.info(f"   Phone: +91 {cleaned_phone}")
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Registration error: {e}")
            flash('An error occurred during registration. Please try again.', 'error')
    
    return render_template('auth/register.html')


@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page"""
    total_detections = DiseaseDetection.query.filter_by(user_id=current_user.id).count()
    
    recent_detections = DiseaseDetection.query.filter_by(user_id=current_user.id)\
        .order_by(DiseaseDetection.detection_time.desc()).limit(10).all()
    
    recent_logins = LoginHistory.query.filter_by(user_id=current_user.id)\
        .order_by(LoginHistory.login_time.desc()).limit(5).all()
    
    # Session info
    session_info = {
        'session_start': session.get('session_start'),
        'server_start': session.get('server_start'),
        'is_valid': 'session_start' in session
    }
    
    return render_template('auth/profile.html',
        total_detections=total_detections,
        recent_detections=recent_detections,
        recent_logins=recent_logins,
        session_info=session_info
    )


@auth_bp.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit user profile with 10-digit phone validation"""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        location = request.form.get('location', '').strip()
        farm_size = request.form.get('farm_size', '')
        farm_size_unit = request.form.get('farm_size_unit', 'hectare')
        
        errors = []
        
        # Email validation
        if not email or '@' not in email:
            errors.append('Valid email is required')
        
        # Check if email is taken by another user
        existing_user = User.query.filter_by(email=email).first()
        if existing_user and existing_user.id != current_user.id:
            errors.append('Email already in use by another account')
        
        # ✅ PHONE NUMBER VALIDATION (10 digits)
        phone_valid, cleaned_phone, phone_error = validate_phone_number(phone)
        if not phone_valid:
            errors.append(phone_error)
        
        # Display errors
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('auth/edit_profile.html')
        
        try:
            # Update user profile
            current_user.full_name = full_name
            current_user.email = email
            current_user.phone = cleaned_phone  # ✅ Store cleaned 10-digit phone
            current_user.location = location
            current_user.farm_size = float(farm_size) if farm_size else None
            current_user.farm_size_unit = farm_size_unit
            
            db.session.commit()
            
            logger.info(f"✅ Profile updated: {current_user.username}")
            if cleaned_phone:
                logger.info(f"   New phone: +91 {cleaned_phone}")
            
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('auth.profile'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Profile update error: {e}")
            flash('Error updating profile. Please try again.', 'error')
    
    return render_template('auth/edit_profile.html')


@auth_bp.route('/profile/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validate current password
        if not current_user.check_password(current_password):
            flash('Current password is incorrect', 'error')
            return render_template('auth/change_password.html')
        
        # Validate new password
        if len(new_password) < 6:
            flash('New password must be at least 6 characters', 'error')
            return render_template('auth/change_password.html')
        
        # Check password match
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return render_template('auth/change_password.html')
        
        # Check if new password is same as current
        if current_password == new_password:
            flash('New password must be different from current password', 'error')
            return render_template('auth/change_password.html')
        
        try:
            # Update password
            current_user.set_password(new_password)
            db.session.commit()
            
            logger.info(f"✅ Password changed: {current_user.username}")
            
            flash('Password changed successfully!', 'success')
            return redirect(url_for('auth.profile'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"❌ Password change error: {e}")
            flash('Error changing password. Please try again.', 'error')
    
    return render_template('auth/change_password.html')


@auth_bp.route('/check-session')
@login_required
def check_session():
    """API endpoint to check session validity"""
    return jsonify({
        'authenticated': current_user.is_authenticated,
        'username': current_user.username,
        'session_start': session.get('session_start'),
        'server_start': session.get('server_start'),
        'current_server_start': current_app.config.get('SERVER_START_TIME'),
        'session_valid': session.get('server_start') == current_app.config.get('SERVER_START_TIME')
    })


@auth_bp.route('/api/user-stats')
@login_required
def user_stats_api():
    """API endpoint for user statistics"""
    total_detections = DiseaseDetection.query.filter_by(user_id=current_user.id).count()
    
    # Disease breakdown
    disease_counts = db.session.query(
        DiseaseDetection.detected_disease,
        db.func.count(DiseaseDetection.id)
    ).filter_by(user_id=current_user.id)\
     .group_by(DiseaseDetection.detected_disease)\
     .all()
    
    return jsonify({
        'username': current_user.username,
        'total_detections': total_detections,
        'login_count': current_user.login_count,
        'member_since': current_user.created_at.isoformat(),
        'last_login': current_user.last_login.isoformat() if current_user.last_login else None,
        'disease_breakdown': [{'disease': d, 'count': c} for d, c in disease_counts],
        'phone': f"+91 {current_user.phone}" if current_user.phone else None
    })


@auth_bp.route('/api/validate-username/<username>')
def validate_username(username):
    """API endpoint to check if username is available"""
    if len(username) < 3:
        return jsonify({
            'available': False,
            'message': 'Username must be at least 3 characters'
        })
    
    existing_user = User.query.filter_by(username=username).first()
    
    return jsonify({
        'available': existing_user is None,
        'message': 'Username available' if existing_user is None else 'Username already taken'
    })


@auth_bp.route('/api/validate-email/<email>')
def validate_email(email):
    """API endpoint to check if email is available"""
    if '@' not in email:
        return jsonify({
            'available': False,
            'message': 'Invalid email format'
        })
    
    existing_user = User.query.filter_by(email=email).first()
    
    return jsonify({
        'available': existing_user is None,
        'message': 'Email available' if existing_user is None else 'Email already registered'
    })


@auth_bp.route('/api/validate-phone', methods=['POST'])
def validate_phone_api():
    """API endpoint to validate phone number (10 digits)"""
    data = request.get_json()
    phone = data.get('phone', '')
    
    is_valid, cleaned_phone, error_message = validate_phone_number(phone)
    
    return jsonify({
        'valid': is_valid,
        'cleaned_phone': cleaned_phone,
        'formatted_phone': f"+91 {cleaned_phone}" if cleaned_phone else None,
        'message': error_message if not is_valid else 'Valid 10 digit number'
    })


@auth_bp.route('/delete-account', methods=['POST'])
@login_required
def delete_account():
    """Delete user account (with confirmation)"""
    password = request.form.get('password', '')
    confirm_text = request.form.get('confirm_text', '').strip()
    
    # Verify password
    if not current_user.check_password(password):
        flash('Incorrect password. Account deletion cancelled.', 'error')
        return redirect(url_for('auth.profile'))
    
    # Verify confirmation text
    if confirm_text.upper() != 'DELETE':
        flash('Incorrect confirmation text. Account deletion cancelled.', 'error')
        return redirect(url_for('auth.profile'))
    
    try:
        username = current_user.username
        user_id = current_user.id
        
        # Delete user (cascades to login_history and disease_detections)
        db.session.delete(current_user)
        db.session.commit()
        
        # Logout
        session.clear()
        logout_user()
        
        logger.info(f"🗑️ Account deleted: {username} (ID: {user_id})")
        
        flash('Your account has been permanently deleted.', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"❌ Account deletion error: {e}")
        flash('Error deleting account. Please contact support.', 'error')
        return redirect(url_for('auth.profile'))


# Error handlers for auth blueprint
@auth_bp.errorhandler(404)
def auth_not_found(error):
    """Handle 404 errors in auth routes"""
    flash('Page not found', 'error')
    return redirect(url_for('auth.login'))


@auth_bp.errorhandler(500)
def auth_server_error(error):
    """Handle 500 errors in auth routes"""
    db.session.rollback()
    logger.error(f"Server error in auth: {error}")
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('auth.login'))