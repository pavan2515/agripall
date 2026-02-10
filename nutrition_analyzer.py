import cv2
import numpy as np
import logging
from PIL import Image
import json
import os
import time

logger = logging.getLogger(__name__)

# Load nutrition deficiency database
def load_nutrition_deficiency_data():
    """Load nutrition deficiency information from JSON"""
    try:
        nutrition_path = 'nutrition_deficiency.json'
        if os.path.exists(nutrition_path):
            with open(nutrition_path, 'r') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded {len(data)} nutrition deficiency types")
                return data
        else:
            logger.error(f"Nutrition deficiency file not found at: {os.path.abspath(nutrition_path)}")
            return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading nutrition deficiency data: {e}")
        return {}


def remove_background_balanced(image):
    """
    BALANCED: Fast but accurate background removal
    - Resize for speed (but not too small)
    - 3 iterations (middle ground)
    """
    try:
        # Resize if too large (SPEED OPTIMIZATION)
        h, w = image.shape[:2]
        max_size = 1000  # Slightly larger than fast version for better accuracy
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized: {w}x{h} → {new_w}x{new_h}")
        else:
            img = image.copy()
        
        h, w = img.shape[:2]
        
        # Create mask
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Rectangle around leaf
        margin = int(min(h, w) * 0.08)
        rect = (margin, margin, w - 2*margin, h - 2*margin)
        
        # BALANCED: 3 iterations (not 2, not 5)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        
        # Binary mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # White background
        white_bg = np.ones_like(img) * 255
        segmented = np.where(mask2[:, :, None] == 1, img, white_bg)
        
        # Resize back if needed
        if max(image.shape[:2]) > max_size:
            segmented = cv2.resize(segmented, (image.shape[1], image.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
            mask2 = cv2.resize(mask2, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        logger.info("✅ Background removed")
        return segmented, mask2
        
    except Exception as e:
        logger.error(f"❌ Error removing background: {e}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        return image, mask


def analyze_leaf_color_patterns(image):
    """BALANCED: Analyze color patterns"""
    try:
        # Background removal
        segmented_image, leaf_mask = remove_background_balanced(image)
        
        # Convert to color spaces
        hsv = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        
        # Calculate stats
        h, s, v = cv2.split(hsv)
        
        h_mean = cv2.mean(h, mask=leaf_mask)[0]
        s_mean = cv2.mean(s, mask=leaf_mask)[0]
        v_mean = cv2.mean(v, mask=leaf_mask)[0]
        
        # All pattern detections
        patterns = {
            'yellowing': detect_yellowing(hsv, leaf_mask),
            'purpling': detect_purpling(hsv, rgb, leaf_mask),
            'interveinal_chlorosis': detect_interveinal_chlorosis_fast(segmented_image, leaf_mask),
            'marginal_chlorosis': detect_marginal_chlorosis_fast(segmented_image, leaf_mask),
            'pale_color': detect_pale_color(hsv, leaf_mask),
            'necrosis': detect_necrosis(hsv, leaf_mask),
            'bleaching': detect_bleaching(hsv, leaf_mask)
        }
        
        return {
            'color_stats': {
                'hue_mean': h_mean,
                'saturation_mean': s_mean,
                'value_mean': v_mean,
                'a_channel': 0,
                'b_channel': 0
            },
            'patterns': patterns,
            'segmented_image': segmented_image
        }
        
    except Exception as e:
        logger.error(f"Error analyzing leaf: {e}")
        return None


def detect_yellowing(hsv, leaf_mask):
    """Detect yellow coloration"""
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([40, 255, 255])
    
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.bitwise_and(yellow_mask, leaf_mask)
    
    leaf_area = cv2.countNonZero(leaf_mask)
    yellow_area = cv2.countNonZero(yellow_mask)
    
    percentage = (yellow_area / leaf_area * 100) if leaf_area > 0 else 0
    
    return {
        'detected': percentage > 10,
        'severity': 'high' if percentage > 40 else 'moderate' if percentage > 20 else 'mild',
        'percentage': round(percentage, 2)
    }


def detect_purpling(hsv, rgb, leaf_mask):
    """Detect purple/red tinting"""
    lower_purple1 = np.array([140, 30, 50])
    upper_purple1 = np.array([180, 255, 255])
    lower_purple2 = np.array([0, 30, 50])
    upper_purple2 = np.array([10, 255, 255])
    
    purple_mask1 = cv2.inRange(hsv, lower_purple1, upper_purple1)
    purple_mask2 = cv2.inRange(hsv, lower_purple2, upper_purple2)
    purple_mask = cv2.bitwise_or(purple_mask1, purple_mask2)
    purple_mask = cv2.bitwise_and(purple_mask, leaf_mask)
    
    leaf_area = cv2.countNonZero(leaf_mask)
    purple_area = cv2.countNonZero(purple_mask)
    
    percentage = (purple_area / leaf_area * 100) if leaf_area > 0 else 0
    
    return {
        'detected': percentage > 5,
        'severity': 'high' if percentage > 20 else 'moderate' if percentage > 10 else 'mild',
        'percentage': round(percentage, 2)
    }


def detect_interveinal_chlorosis_fast(image, leaf_mask):
    """FAST interveinal chlorosis detection"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simplified edge detection
        edges = cv2.Canny(gray, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        veins = cv2.dilate(edges, kernel, iterations=1)
        
        inter_vein = cv2.bitwise_not(veins)
        inter_vein = cv2.bitwise_and(inter_vein, leaf_mask)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        
        inter_vein_brightness = cv2.mean(v, mask=inter_vein)[0]
        vein_brightness = cv2.mean(v, mask=veins)[0]
        
        brightness_diff = inter_vein_brightness - vein_brightness
        
        detected = brightness_diff > 15
        severity = 'high' if brightness_diff > 40 else 'moderate' if brightness_diff > 25 else 'mild'
        
        return {
            'detected': detected,
            'severity': severity,
            'brightness_difference': round(brightness_diff, 2)
        }
    except:
        return {'detected': False, 'severity': 'none', 'brightness_difference': 0}


def detect_marginal_chlorosis_fast(image, leaf_mask):
    """FAST marginal chlorosis detection"""
    try:
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'detected': False, 'severity': 'none', 'percentage': 0}
        
        margin_mask = np.zeros_like(leaf_mask)
        for contour in contours:
            cv2.drawContours(margin_mask, [contour], -1, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
            inner = cv2.erode(margin_mask.copy(), kernel, iterations=1)
            margin_mask = cv2.subtract(margin_mask, inner)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow_brown = np.array([15, 30, 50])
        upper_yellow_brown = np.array([35, 255, 255])
        
        yellow_brown_mask = cv2.inRange(hsv, lower_yellow_brown, upper_yellow_brown)
        margin_affected = cv2.bitwise_and(yellow_brown_mask, margin_mask)
        
        margin_area = cv2.countNonZero(margin_mask)
        affected_area = cv2.countNonZero(margin_affected)
        
        percentage = (affected_area / margin_area * 100) if margin_area > 0 else 0
        
        return {
            'detected': percentage > 15,
            'severity': 'high' if percentage > 50 else 'moderate' if percentage > 30 else 'mild',
            'percentage': round(percentage, 2)
        }
    except:
        return {'detected': False, 'severity': 'none', 'percentage': 0}


def detect_pale_color(hsv, leaf_mask):
    """Detect overall pale color"""
    s = hsv[:, :, 1]
    mean_saturation = cv2.mean(s, mask=leaf_mask)[0]
    
    is_pale = mean_saturation < 80
    severity = 'high' if mean_saturation < 50 else 'moderate' if mean_saturation < 70 else 'mild'
    
    return {
        'detected': is_pale,
        'severity': severity,
        'saturation_level': round(mean_saturation, 2)
    }


def detect_necrosis(hsv, leaf_mask):
    """Detect necrotic tissue"""
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([25, 255, 150])
    
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_mask = cv2.bitwise_and(brown_mask, leaf_mask)
    
    leaf_area = cv2.countNonZero(leaf_mask)
    brown_area = cv2.countNonZero(brown_mask)
    
    percentage = (brown_area / leaf_area * 100) if leaf_area > 0 else 0
    
    return {
        'detected': percentage > 5,
        'severity': 'high' if percentage > 25 else 'moderate' if percentage > 10 else 'mild',
        'percentage': round(percentage, 2)
    }


def detect_bleaching(hsv, leaf_mask):
    """Detect bleached areas"""
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    
    high_v = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
    low_s = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY_INV)[1]
    
    bleached_mask = cv2.bitwise_and(high_v, low_s)
    bleached_mask = cv2.bitwise_and(bleached_mask, leaf_mask)
    
    leaf_area = cv2.countNonZero(leaf_mask)
    bleached_area = cv2.countNonZero(bleached_mask)
    
    percentage = (bleached_area / leaf_area * 100) if leaf_area > 0 else 0
    
    return {
        'detected': percentage > 3,
        'severity': 'high' if percentage > 15 else 'moderate' if percentage > 8 else 'mild',
        'percentage': round(percentage, 2)
    }


def diagnose_nutrient_deficiency(color_analysis):
    """
    IMPROVED: More flexible diagnosis
    """
    patterns = color_analysis['patterns']
    
    diagnoses = []
    
    # Nitrogen deficiency - RELAXED (detects your 51.58% yellowing!)
    if patterns['yellowing']['detected'] or patterns['pale_color']['detected']:
        if patterns['yellowing']['detected']:
            confidence = min(patterns['yellowing']['percentage'] * 1.5, 95)
        else:
            confidence = 60
        
        diagnoses.append({
            'deficiency': 'Nitrogen_Deficiency',
            'confidence': round(confidence, 2),
            'primary_symptoms': ['Overall yellowing', 'Pale green color', 'Stunted growth'],
            'affected_area': 'Older leaves (lower leaves affected first)'
        })
    
    # Phosphorus deficiency
    if patterns['purpling']['detected']:
        confidence = min(patterns['purpling']['percentage'] * 2.5, 90)
        diagnoses.append({
            'deficiency': 'Phosphorus_Deficiency',
            'confidence': round(confidence, 2),
            'primary_symptoms': ['Purple/reddish coloration', 'Dark green leaves'],
            'affected_area': 'Lower leaves and stems'
        })
    
    # Potassium deficiency
    if patterns['marginal_chlorosis']['detected'] or (patterns['necrosis']['detected'] and patterns['yellowing']['detected']):
        confidence = 70 if patterns['marginal_chlorosis']['detected'] else 60
        diagnoses.append({
            'deficiency': 'Potassium_Deficiency',
            'confidence': round(confidence, 2),
            'primary_symptoms': ['Yellowing and browning of leaf edges', 'Leaf tip burn'],
            'affected_area': 'Leaf margins and tips (older leaves first)'
        })
    
    # Magnesium deficiency
    if patterns['interveinal_chlorosis']['detected'] and patterns['yellowing']['detected']:
        confidence = 75
        diagnoses.append({
            'deficiency': 'Magnesium_Deficiency',
            'confidence': round(confidence, 2),
            'primary_symptoms': ['Yellowing between veins', 'Veins remain green'],
            'affected_area': 'Older leaves (interveinal areas)'
        })
    
    # Iron deficiency
    if patterns['interveinal_chlorosis']['detected'] or patterns['bleaching']['detected']:
        confidence = 80 if patterns['bleaching']['detected'] else 65
        diagnoses.append({
            'deficiency': 'Iron_Deficiency',
            'confidence': round(confidence, 2),
            'primary_symptoms': ['Yellowing between veins on new growth', 'White/bleached leaves'],
            'affected_area': 'Young leaves (newest growth)'
        })
    
    # Sort by confidence
    diagnoses = sorted(diagnoses, key=lambda x: x['confidence'], reverse=True)
    
    return diagnoses


def analyze_nutrition_deficiency(image_path):
    """
    BALANCED: Fast (10-15s) AND Accurate
    """
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("⚖️ BALANCED NUTRITION ANALYSIS (Fast + Accurate)")
    logger.info("="*80)
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"📸 Analyzing: {image_path}")
        
        color_analysis = analyze_leaf_color_patterns(image)
        
        if not color_analysis:
            return {'success': False, 'error': 'Failed to analyze'}
        
        diagnoses = diagnose_nutrient_deficiency(color_analysis)
        
        nutrition_data = load_nutrition_deficiency_data()
        
        detailed_results = []
        for diagnosis in diagnoses[:3]:
            deficiency_key = diagnosis['deficiency']
            
            if deficiency_key in nutrition_data:
                deficiency_info = nutrition_data[deficiency_key]
                
                detailed_results.append({
                    'deficiency': deficiency_key,
                    'name': deficiency_info['name'],
                    'nutrient': deficiency_info['nutrient'],
                    'confidence': diagnosis['confidence'],
                    'primary_symptoms': diagnosis['primary_symptoms'],
                    'affected_area': diagnosis['affected_area'],
                    'description': deficiency_info['description'],
                    'visual_symptoms': deficiency_info['visual_symptoms'],
                    'severity_indicators': deficiency_info['severity_indicators'],
                    'treatment': deficiency_info['treatment'],
                    'fertilizer': deficiency_info['fertilizer'],
                    'prevention': deficiency_info['prevention']
                })
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Complete in {elapsed:.2f}s - Found {len(detailed_results)} deficiencies")
        
        return {
            'success': True,
            'color_analysis': color_analysis,
            'diagnoses': detailed_results,
            'total_found': len(detailed_results),
            'processing_time': round(elapsed, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}


def calculate_fertilizer_dosage(area, area_unit, fertilizer_info):
    """Calculate fertilizer dosage"""
    conversion_factors = {
        'square_meter': 0.0001,
        'acre': 0.404686,
        'hectare': 1.0,
        'square_foot': 0.0000092903
    }
    
    area_in_hectares = area * conversion_factors.get(area_unit, 1.0)
    
    chemical_dosage = {
        'amount': round(fertilizer_info['chemical']['dosage_per_hectare'] * area_in_hectares, 2),
        'unit': fertilizer_info['chemical']['unit'],
        'name': fertilizer_info['chemical']['name']
    }
    
    organic_dosage = {
        'amount': round(fertilizer_info['organic']['dosage_per_hectare'] * area_in_hectares, 2),
        'unit': fertilizer_info['organic']['unit'],
        'name': fertilizer_info['organic']['name']
    }
    
    return chemical_dosage, organic_dosage, area_in_hectares