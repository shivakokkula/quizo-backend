# app.py - Complete Standalone Meme Generator
from flask import Flask, request, jsonify, send_file, render_template_string
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import requests
from datetime import datetime, timedelta
import uuid
import json
import sqlite3
import hashlib
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'meme-generator-secret-key-2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
DATABASE_FILE = 'meme_generator.db'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# Popular meme templates (no AI generation - just predefined templates)
MEME_TEMPLATES = {
    "drake": {
        "name": "Drake Pointing",
        "url": "https://i.imgflip.com/30b1gx.jpg",
        "width": 1200,
        "height": 1200,
        "text_areas": [
            {"x": 600, "y": 200, "width": 550, "height": 200, "align": "left"},
            {"x": 600, "y": 800, "width": 550, "height": 200, "align": "left"}
        ]
    },
    "distracted_boyfriend": {
        "name": "Distracted Boyfriend",
        "url": "https://i.imgflip.com/1ur9b0.jpg",
        "width": 1200, 
        "height": 800,
        "text_areas": [
            {"x": 150, "y": 50, "width": 150, "height": 80, "align": "center"},
            {"x": 500, "y": 50, "width": 200, "height": 80, "align": "center"},
            {"x": 900, "y": 50, "width": 150, "height": 80, "align": "center"}
        ]
    },
    "two_buttons": {
        "name": "Two Buttons",
        "url": "https://i.imgflip.com/1g8my4.jpg", 
        "width": 1200,
        "height": 908,
        "text_areas": [
            {"x": 220, "y": 160, "width": 180, "height": 80, "align": "center"},
            {"x": 600, "y": 160, "width": 180, "height": 80, "align": "center"}
        ]
    },
    "woman_yelling_cat": {
        "name": "Woman Yelling at Cat",
        "url": "https://i.imgflip.com/345v97.jpg",
        "width": 1200,
        "height": 800,
        "text_areas": [
            {"x": 100, "y": 50, "width": 400, "height": 100, "align": "center"},
            {"x": 700, "y": 50, "width": 400, "height": 100, "align": "center"}
        ]
    },
    "change_my_mind": {
        "name": "Change My Mind",
        "url": "https://i.imgflip.com/24y43o.jpg",
        "width": 1200,
        "height": 900,
        "text_areas": [
            {"x": 200, "y": 400, "width": 600, "height": 200, "align": "center"}
        ]
    }
}

# Subscription tiers for monetization
SUBSCRIPTION_TIERS = {
    'free': {
        'name': 'Free',
        'price': 0,
        'daily_limit': 10,
        'features': ['Basic memes', 'Standard templates', 'Watermark'],
        'watermark': True,
        'hd_quality': False
    },
    'pro': {
        'name': 'Pro',
        'price': 9.99,
        'daily_limit': 500,
        'features': ['HD memes', 'All templates', 'No watermark', 'API access'],
        'watermark': False,
        'hd_quality': True
    },
    'enterprise': {
        'name': 'Enterprise',
        'price': 49.99,
        'daily_limit': 10000,
        'features': ['Unlimited HD memes', 'Custom templates', 'White-label', 'Priority support'],
        'watermark': False,
        'hd_quality': True
    }
}

class DatabaseManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                api_key TEXT UNIQUE,
                subscription_tier TEXT DEFAULT 'free',
                daily_usage INTEGER DEFAULT 0,
                last_usage_reset DATE DEFAULT CURRENT_DATE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Memes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                meme_id TEXT UNIQUE,
                filename TEXT,
                top_text TEXT,
                bottom_text TEXT,
                template_used TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                downloads INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # API usage tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                endpoint TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, email, password):
        """Create new user"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        password_hash = generate_password_hash(password)
        api_key = self.generate_api_key()
        
        try:
            cursor.execute('''
                INSERT INTO users (email, password_hash, api_key)
                VALUES (?, ?, ?)
            ''', (email, password_hash, api_key))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return user_id, api_key
        except sqlite3.IntegrityError:
            conn.close()
            return None, None
    
    def authenticate_user(self, email, password):
        """Authenticate user"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash, api_key FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[1], password):
            return {'id': user[0], 'api_key': user[2]}
        return None
    
    def get_user_by_api_key(self, api_key):
        """Get user by API key"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, email, subscription_tier, daily_usage, last_usage_reset 
            FROM users WHERE api_key = ?
        ''', (api_key,))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'email': user[1],
                'subscription_tier': user[2],
                'daily_usage': user[3],
                'last_usage_reset': user[4]
            }
        return None
    
    def update_daily_usage(self, user_id):
        """Update daily usage counter"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Reset counter if new day
        cursor.execute('''
            UPDATE users SET 
                daily_usage = CASE 
                    WHEN last_usage_reset < DATE('now') THEN 1
                    ELSE daily_usage + 1
                END,
                last_usage_reset = CASE
                    WHEN last_usage_reset < DATE('now') THEN DATE('now')
                    ELSE last_usage_reset
                END
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def save_meme(self, user_id, meme_data):
        """Save meme metadata"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO memes (user_id, meme_id, filename, top_text, bottom_text, template_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            meme_data.get('meme_id'),
            meme_data.get('filename'),
            meme_data.get('top_text', ''),
            meme_data.get('bottom_text', ''),
            meme_data.get('template', '')
        ))
        
        conn.commit()
        conn.close()
    
    def generate_api_key(self):
        """Generate unique API key"""
        return 'mk_' + hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:32]

class MemeGenerator:
    def __init__(self):
        self.font_sizes = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80]
    
    def get_font(self, size=32):
        """Get font with fallbacks"""
        font_paths = [
            # Windows
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            # macOS
            "/System/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # Generic paths
            "arial.ttf",
            "Arial.ttf"
        ]
        
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
        
        # Fallback to default font
        try:
            return ImageFont.load_default()
        except:
            return None
    
    def add_text_with_outline(self, draw, text, position, font, fill_color="white", outline_color="black", outline_width=2):
        """Add text with black outline"""
        if not font:
            return
            
        x, y = position
        
        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=fill_color)
    
    def wrap_text(self, text, font, max_width):
        """Wrap text to fit width"""
        if not font:
            return [text]
            
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = font.getbbox(test_line)
                width = bbox[2] - bbox[0]
            except:
                # Fallback for older Pillow versions
                width = font.getsize(test_line)[0]
            
            if width <= max_width or not current_line:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def fit_text_to_area(self, text, width, height):
        """Find best font size for text area"""
        for size in reversed(self.font_sizes):
            font = self.get_font(size)
            if not font:
                continue
                
            lines = self.wrap_text(text, font, width * 0.9)
            
            try:
                line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
            except:
                line_height = font.getsize('A')[1]
                
            total_height = len(lines) * line_height
            
            if total_height <= height * 0.9:
                return font, lines
        
        # Fallback
        font = self.get_font(16)
        lines = self.wrap_text(text, font, width * 0.9) if font else [text]
        return font, lines
    
    def add_watermark(self, img):
        """Add watermark for free users"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        watermark_text = "MemeGen Pro"
        font_size = max(12, width // 80)
        font = self.get_font(font_size)
        
        if font:
            try:
                bbox = font.getbbox(watermark_text)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width, text_height = font.getsize(watermark_text)
            
            # Bottom right corner
            x = width - text_width - 10
            y = height - text_height - 10
            
            # Semi-transparent background
            padding = 3
            draw.rectangle([x-padding, y-padding, x+text_width+padding, y+text_height+padding], 
                          fill=(0, 0, 0, 100))
            
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 200))
        
        return img
    
    def generate_basic_meme(self, top_text="", bottom_text="", template_url=None, 
                           custom_image=None, add_watermark=True, hd_quality=False):
        """Generate basic top/bottom meme"""
        try:
            # Load base image
            if custom_image:
                img = Image.open(custom_image)
            elif template_url:
                try:
                    response = requests.get(template_url, timeout=10)
                    response.raise_for_status()
                    img = Image.open(io.BytesIO(response.content))
                except:
                    # Fallback to blank image
                    img = Image.new('RGB', (800, 600), color='white')
            else:
                size = (1200, 900) if hd_quality else (800, 600)
                img = Image.new('RGB', size, color='white')
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for HD
            if hd_quality:
                max_size = (1920, 1080)
                img.thumbnail(max_size, Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
            
            draw = ImageDraw.Draw(img)
            width, height = img.size
            
            # Add top text
            if top_text.strip():
                font, lines = self.fit_text_to_area(top_text, width * 0.9, height * 0.25)
                if font and lines:
                    try:
                        line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
                    except:
                        line_height = font.getsize('A')[1]
                    
                    total_height = len(lines) * line_height
                    start_y = max(10, (height * 0.25 - total_height) // 2)
                    
                    for i, line in enumerate(lines):
                        try:
                            bbox = font.getbbox(line)
                            text_width = bbox[2] - bbox[0]
                        except:
                            text_width = font.getsize(line)[0]
                        
                        x = (width - text_width) // 2
                        y = start_y + i * line_height
                        
                        self.add_text_with_outline(draw, line, (x, y), font)
            
            # Add bottom text
            if bottom_text.strip():
                font, lines = self.fit_text_to_area(bottom_text, width * 0.9, height * 0.25)
                if font and lines:
                    try:
                        line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
                    except:
                        line_height = font.getsize('A')[1]
                    
                    total_height = len(lines) * line_height
                    start_y = height - total_height - max(10, (height * 0.25 - total_height) // 2)
                    
                    for i, line in enumerate(lines):
                        try:
                            bbox = font.getbbox(line)
                            text_width = bbox[2] - bbox[0]
                        except:
                            text_width = font.getsize(line)[0]
                        
                        x = (width - text_width) // 2
                        y = start_y + i * line_height
                        
                        self.add_text_with_outline(draw, line, (x, y), font)
            
            # Add watermark if needed
            if add_watermark:
                img = self.add_watermark(img)
            
            return img
            
        except Exception as e:
            print(f"Error generating meme: {e}")
            return None
    
    def generate_template_meme(self, template_name, texts, add_watermark=True, hd_quality=False):
        """Generate meme using template"""
        try:
            if template_name not in MEME_TEMPLATES:
                return None
            
            template = MEME_TEMPLATES[template_name]
            
            # Download template image
            try:
                response = requests.get(template['url'], timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
            except:
                # Fallback to blank image
                img = Image.new('RGB', (template['width'], template['height']), color='lightgray')
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to template dimensions
            img = img.resize((template['width'], template['height']))
            
            draw = ImageDraw.Draw(img)
            
            # Add text to each area
            for i, text_area in enumerate(template['text_areas']):
                if i < len(texts) and texts[i] and texts[i].strip():
                    text = texts[i].strip()
                    
                    # Get font and wrapped text
                    font, lines = self.fit_text_to_area(text, text_area['width'], text_area['height'])
                    
                    if font and lines:
                        try:
                            line_height = font.getbbox('A')[3] - font.getbbox('A')[1]
                        except:
                            line_height = font.getsize('A')[1]
                        
                        total_height = len(lines) * line_height
                        start_y = text_area['y'] + (text_area['height'] - total_height) // 2
                        
                        for j, line in enumerate(lines):
                            try:
                                bbox = font.getbbox(line)
                                text_width = bbox[2] - bbox[0]
                            except:
                                text_width = font.getsize(line)[0]
                            
                            # Position based on alignment
                            if text_area['align'] == 'center':
                                x = text_area['x'] + (text_area['width'] - text_width) // 2
                            elif text_area['align'] == 'right':
                                x = text_area['x'] + text_area['width'] - text_width
                            else:  # left
                                x = text_area['x']
                            
                            y = start_y + j * line_height
                            
                            self.add_text_with_outline(draw, line, (x, y), font)
            
            # Add watermark if needed
            if add_watermark:
                img = self.add_watermark(img)
            
            return img
            
        except Exception as e:
            print(f"Error generating template meme: {e}")
            return None

# Initialize components
db = DatabaseManager()
meme_gen = MemeGenerator()

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'API key required', 'code': 'NO_API_KEY'}), 401
        
        user = db.get_user_by_api_key(api_key)
        if not user:
            return jsonify({'error': 'Invalid API key', 'code': 'INVALID_API_KEY'}), 401
        
        # Check daily limits
        tier = SUBSCRIPTION_TIERS[user['subscription_tier']]
        if user['daily_usage'] >= tier['daily_limit']:
            return jsonify({
                'error': 'Daily limit exceeded', 
                'code': 'LIMIT_EXCEEDED',
                'limit': tier['daily_limit'],
                'current_usage': user['daily_usage']
            }), 429
        
        request.current_user = user
        return f(*args, **kwargs)
    
    return decorated_function

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MemeGen Pro - Professional Meme Generator</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; color: #333;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .card { 
            background: white; padding: 30px; border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 30px;
        }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 600; color: #555; }
        input, textarea, select { 
            width: 100%; padding: 12px; border: 2px solid #ddd; border-radius: 8px; 
            font-size: 14px; transition: border-color 0.3s ease;
        }
        input:focus, textarea:focus, select:focus { 
            border-color: #4CAF50; outline: none; 
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
        }
        button, .btn { 
            background: linear-gradient(45deg, #4CAF50, #45a049); color: white; 
            padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; 
            font-size: 16px; margin: 5px; text-decoration: none; display: inline-block;
            transition: all 0.3s ease;
        }
        button:hover, .btn:hover { 
            transform: translateY(-2px); box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        .template-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin: 20px 0;
        }
        .template-card { 
            border: 2px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; 
            cursor: pointer; transition: all 0.3s ease;
        }
        .template-card:hover { border-color: #4CAF50; transform: translateY(-2px); }
        .template-card.selected { border-color: #4CAF50; background: #f0f8f0; }
        #result { margin-top: 30px; text-align: center; }
        .auth-tabs { display: flex; margin-bottom: 20px; }
        .auth-tab { 
            flex: 1; padding: 10px; background: #e9ecef; border: none; cursor: pointer; 
            border-radius: 5px 5px 0 0; transition: background 0.3s;
        }
        .auth-tab.active { background: #4CAF50; color: white; }
        .auth-section { display: none; background: #f8f9fa; padding: 20px; border-radius: 0 0 10px 10px; }
        .pricing-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 30px; margin: 30px 0;
        }
        .pricing-card { 
            background: white; border-radius: 15px; padding: 30px; text-align: center; 
            box-shadow: 0 5px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease;
        }
        .pricing-card:hover { transform: translateY(-5px); }
        .pricing-card.featured { border: 3px solid #4CAF50; transform: scale(1.05); }
        .price { font-size: 2.5em; font-weight: bold; color: #4CAF50; margin: 20px 0; }
        .features { list-style: none; text-align: left; }
        .features li { padding: 8px 0; border-bottom: 1px solid #eee; }
        .features li:before { content: "✓"; color: #4CAF50; font-weight: bold; margin-right: 10px; }
        .api-docs { background: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px; }
        .code-block { 
            background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; 
            margin: 10px 0; overflow-x: auto; font-family: 'Courier New', monospace;
        }
        @media (max-width: 768px) { 
            .container { padding: 10px; } 
            .header h1 { font-size: 2em; } 
            .pricing-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MemeGen Pro</h1>
            <p>Professional Meme Generator - No AI, Pure Code</p>
        </div>

        <!-- Pricing Plans -->
        <div class="card">
            <h2 style="text-align: center; margin-bottom: 30px;">Choose Your Plan</h2>
            <div class="pricing-grid">
                <div class="pricing-card">
                    <h3>Free</h3>
                    <div class="price">$0<small>/month</small></div>
                    <ul class="features">
                        <li>10 memes per day</li>
                        <li>Basic templates</li>
                        <li>Standard quality</li>
                        <li>Watermarked</li>
                    </ul>
                    <button onclick="selectPlan('free')">Get Started</button>
                </div>
                
                <div class="pricing-card featured">
                    <h3>Pro</h3>
                    <div class="price">$9.99<small>/month</small></div>
                    <ul class="features">
                        <li>500 memes per day</li>
                        <li>All templates</li>
                        <li>HD quality</li>
                        <li>No watermark</li>
                        <li>API access</li>
                    </ul>
                    <button onclick="selectPlan('pro')">Upgrade to Pro</button>
                </div>
                
                <div class="pricing-card">
                    <h3>Enterprise</h3>
                    <div class="price">$49.99<small>/month</small></div>
                    <ul class="features">
                        <li>Unlimited memes</li>
                        <li>Custom templates</li>
                        <li>White-label option</li>
                        <li>Priority support</li>
                        <li>Advanced API</li>
                    </ul>
                    <button onclick="selectPlan('enterprise')">Contact Sales</button>
                </div>
            </div>
        </div>

        <!-- Authentication -->
        <div class="card">
            <div class="auth-tabs">
                <button class="auth-tab active" onclick="showAuthTab('login')">Login</button>
                <button class="auth-tab" onclick="showAuthTab('register')">Register</button>
            </div>
            
            <div id="loginSection" class="auth-section active">
                <h2>Login</h2>
                <form id="loginForm">
                    <input type="email" placeholder="Email" required>
                    <input type="password" placeholder="Password" required>
                    <button type="submit">Login</button>
                </form>
            </div>
            
            <div id="registerSection" class="auth-section">
                <h2>Register</h2>
                <form id="registerForm">
                    <input type="email" placeholder="Email" required>
                    <input type="password" placeholder="Password" required>
                    <button type="submit">Register</button>
                </form>
            </div>
        </div>
    </div>
</body>
</html>
"""