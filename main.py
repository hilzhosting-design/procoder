import os
import json
import sys
from datetime import datetime, timedelta
import pytz
import requests, re
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import itertools
from functools import wraps
import atexit
import logging
import time
import io
import base64

# ML & Data Handling Imports
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Google Cloud Imports
from google.cloud import storage
from google.api_core.exceptions import NotFound

# Set logging level to DEBUG to capture all detailed messages
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

from apscheduler.schedulers.background import BackgroundScheduler

import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth_module
from flask import Flask, render_template, request, flash, redirect, url_for, session, has_request_context, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_client import DocumentSnapshot

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Environment Variables & Config
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" if TELEGRAM_BOT_TOKEN else None
INITIAL_ADMIN_EMAIL = os.getenv("INITIAL_ADMIN_EMAIL", "initial_admin@example.com")

db = None
auth = None
harare_tz = pytz.timezone('Africa/Harare')
utc_tz = pytz.utc

def initialize_firebase_admin_sdk():
    """Initializes the Firebase Admin SDK."""
    global db, auth
    cred = None
    try:
        firebase_config_str = os.getenv('__firebase_config')
        if firebase_config_str:
            firebase_config = json.loads(firebase_config_str)
            cred = credentials.Certificate(firebase_config)
            logging.info("Firebase credentials successfully loaded from environment.")
        else:
            logging.warning("__firebase_config not found. Cannot initialize SDKs.")
            return False
    except Exception as e:
        logging.error(f"Failed to load or parse Firebase credentials: {e}")
        return False

    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        auth = firebase_auth_module
        logging.info("Firebase clients successfully initialized.")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize clients with loaded credentials: {e}")
        return False

if not initialize_firebase_admin_sdk():
    logging.critical("SDKs could not be initialized. Application may not function correctly.")

# Client-side Firebase Web App Configuration
FIREBASE_WEB_CLIENT_CONFIG = {
    "apiKey": os.getenv("FIREBASE_WEB_API_KEY", "YOUR_WEB_API_KEY"),
    "authDomain": os.getenv("FIREBASE_WEB_AUTH_DOMAIN", "YOUR_PROJECT_ID.firebaseapp.com"),
    "projectId": os.getenv("FIREBASE_WEB_PROJECT_ID", "YOUR_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_WEB_STORAGE_BUCKET", "YOUR_PROJECT_ID.appspot.com"),
    "messagingSenderId": os.getenv("FIREBASE_WEB_MESSAGING_SENDER_ID", "YOUR_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_WEB_APP_ID", "YOUR_WEB_APP_ID"),
    "measurementId": os.getenv("FIREBASE_WEB_MEASUREMENT_ID", "YOUR_MEASUREMENT_ID")
}
FIREBASE_WEB_CLIENT_CONFIG_JSON = json.dumps(FIREBASE_WEB_CLIENT_CONFIG)

def normalize_to_harare_time(dt_object):
    if dt_object is None: return None
    if hasattr(dt_object, 'to_datetime'): dt_object = dt_object.to_datetime()
    if dt_object.tzinfo is None: return utc_tz.localize(dt_object).astimezone(harare_tz)
    return dt_object.astimezone(harare_tz)

# --- Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'firebase_uid' not in session:
            flash("You need to be logged in to access this page.", "error")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def subscription_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        uid = session.get('view_as_uid', session.get('firebase_uid'))
        if not uid:
            flash("You need to be logged in to access this page.", "error")
            return redirect(url_for('login'))
        user_doc = get_user_doc_ref(uid).get()
        if not user_doc.exists or not user_doc.to_dict().get('is_subscribed', False):
            flash("This feature requires an active subscription.", "error")
            return redirect(url_for('subscribe_info'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('is_admin', False):
            flash("Access denied. Admin privileges required.", "error")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# --- Firestore Path Helpers ---
def get_app_id_for_firestore():
    return os.getenv('__app_id') or os.getenv('RENDER_SERVICE_ID') or 'default-app-id'

def get_public_prediction_doc_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_predictions').document('current_prediction')

def get_public_prediction_history_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_predictions_history')

def get_public_banner_doc_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_settings').document('banner_message')

def get_backtest_results_doc_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_settings').document('backtest_results')

def get_ml_model_doc_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('ml_models').document('prediction_model')

def get_user_doc_ref(firebase_uid):
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('users').document(firebase_uid)

def get_user_predictions_history_ref(firebase_uid):
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('users').document(firebase_uid).collection('predictions_history')

def get_draws_collection_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_draw_results')

# --- Web Scraping and Data Processing ---
def fetch_draws_from_website():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/536'}
    api_key = os.getenv("SCRAPER_API_KEY")
    if not api_key: return [], "SCRAPER_API_KEY environment variable not found."
    target_url = "https://www.comparethelotto.com/za/gosloto-5-50-results"
    api_request_url = f"http://api.scraperapi.com?api_key={api_key}&url={target_url}"
    try:
        response = requests.get(api_request_url, timeout=60, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        historical_results_body = soup.select_one("div#historicalResults > div.card-body")
        if not historical_results_body: return [], "Could not find historical results container."
        raw_draws_with_timestamps = []
        draw_start_points = historical_results_body.find_all('p', class_='text-muted')
        for date_p_tag in draw_start_points:
            full_date_time_text = date_p_tag.get_text(separator=' ').strip()
            clean_date_time_text = re.sub(r'<[^>]+>|&nbsp;|\s*\([^)]*\)|\s+NEW$', ' ', full_date_time_text).strip()
            clean_date_time_text = re.sub(r'^\w{3}\s+', '', clean_date_time_text)
            clean_date_time_text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', clean_date_time_text).strip()
            try:
                naive_timestamp = datetime.strptime(clean_date_time_text, '%d %B %Y %H:%M')
                timestamp = harare_tz.localize(naive_timestamp)
            except ValueError: continue
            main_numbers, bonus_numbers = [], []
            for sibling in date_p_tag.next_siblings:
                if sibling.name in ('p', 'hr'): break
                if sibling.name == 'span':
                    try:
                        num = int(sibling.text.strip())
                        if 'results-number-bonus' in sibling.get('class', []): bonus_numbers.append(num)
                        elif 'results-number' in sibling.get('class', []): main_numbers.append(num)
                    except (ValueError, TypeError): pass
            if len(main_numbers) == 5 and len(bonus_numbers) == 2:
                draw_type = "Morning" if timestamp.hour < 12 else "Evening"
                raw_draws_with_timestamps.append((timestamp, sorted(main_numbers), bonus_numbers[0], bonus_numbers[1], draw_type))
        raw_draws_with_timestamps.sort(key=lambda x: x[0], reverse=True)
        unique_draws, seen_draw_identifiers = [], set()
        for draw in raw_draws_with_timestamps:
            identifier = (draw[0].strftime('%Y-%m-%d'), draw[4])
            if identifier not in seen_draw_identifiers:
                unique_draws.append(draw)
                seen_draw_identifiers.add(identifier)
        return unique_draws, None
    except requests.exceptions.RequestException as e: return [], f"Error fetching data via API: {e}"

def store_draws_to_firestore(draws_data):
    if db is None: return 0
    draws_collection = get_draws_collection_ref()
    inserted_count = 0
    for timestamp, mains, b1, b2, draw_type in draws_data:
        draw_date_str = timestamp.strftime('%Y-%m-%d')
        query = draws_collection.where(filter=FieldFilter('draw_date', '==', draw_date_str)).where(filter=FieldFilter('draw_type', '==', draw_type)).limit(1).get()
        if not list(query):
            try:
                draws_collection.add({'main1': mains[0], 'main2': mains[1], 'main3': mains[2], 'main4': mains[3], 'main5': mains[4], 'bonus1': b1, 'bonus2': b2, 'draw_date': draw_date_str, 'draw_type': draw_type, 'timestamp': timestamp})
                inserted_count += 1
            except Exception as e: logging.error(f"Failed to add draw to Firestore: {e}")
    return inserted_count

def get_historical_draws_from_firestore(limit=None, history_window_days=None):
    if db is None: return []
    draws_collection = get_draws_collection_ref()
    query = draws_collection.order_by('timestamp', direction=firestore.Query.DESCENDING)
    if history_window_days:
        start_date_for_window = datetime.now(harare_tz) - timedelta(days=history_window_days)
        query = query.where(filter=FieldFilter('timestamp', '>=', start_date_for_window))
    if limit: query = query.limit(limit)
    draws = []
    try:
        for doc in query.stream():
            data = doc.to_dict()
            timestamp_dt = normalize_to_harare_time(data.get('timestamp'))
            if not timestamp_dt: continue
            mains = [data.get(f'main{i}') for i in range(1, 6)]
            bonuses = [data.get('bonus1'), data.get('bonus2')]
            if any(m is None for m in mains) or any(b is None for b in bonuses): continue
            draws.append((timestamp_dt, mains, bonuses[0], bonuses[1], data.get('draw_type', 'Unknown')))
    except Exception as e: logging.error(f"Failed to fetch historical draws: {e}")
    return draws

# --- Machine Learning Section ---
def get_following_numbers_pool(base_mains, historical_draws):
    following_numbers = []
    base_mains_set = set(base_mains)
    for i in range(len(historical_draws) - 1):
        current_mains = set(historical_draws[i][1])
        previous_mains = set(historical_draws[i+1][1])
        if not base_mains_set.isdisjoint(previous_mains):
            following_numbers.extend(list(current_mains))
    return [num for num, count in Counter(following_numbers).most_common(10)]

def create_feature_dataset(historical_draws):
    features = []
    if len(historical_draws) < 25:
        return None, None
    for i in range(len(historical_draws) - 20):
        target_draw_mains = set(historical_draws[i][1])
        base_draw = historical_draws[i+1]
        prediction_history = historical_draws[i+2:]
        _, base_mains, b1, b2, _ = base_draw
        following_pool = get_following_numbers_pool(base_mains, prediction_history)
        recency_freqs = Counter(num for draw in prediction_history[:15] for num in draw[1])
        for num in range(1, 51):
            feature_vector = {
                'number': num,
                'is_in_following_pool': 1 if num in following_pool else 0,
                'recency_count_15': recency_freqs.get(num, 0),
                'was_in_prev_mains': 1 if num in base_mains else 0,
                'was_in_prev_bonus': 1 if num in [b1, b2] else 0,
                'was_winner': 1 if num in target_draw_mains else 0
            }
            features.append(feature_vector)
    if not features:
        return None, None
    df = pd.DataFrame(features)
    X = df.drop(['was_winner', 'number'], axis=1)
    y = df['was_winner']
    return X, y

def train_and_save_model():
    logging.info("--- ML_TRAINING: Starting model training process. ---")
    try:
        historical_draws = get_historical_draws_from_firestore(history_window_days=365)
        if len(historical_draws) < 50:
            return False, "Not enough historical draws to train model (need at least 50)."
        
        X, y = create_feature_dataset(historical_draws)
        if X is None:
            return False, "Feature dataset could not be created."
            
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X, y)
        
        model_as_bytes = io.BytesIO()
        joblib.dump(model, model_as_bytes)
        model_as_bytes.seek(0)
        base64_encoded_model = base64.b64encode(model_as_bytes.read()).decode('utf-8')

        get_ml_model_doc_ref().set({
            'model_data_base64': base64_encoded_model,
            'last_trained': firestore.SERVER_TIMESTAMP,
            'training_draw_count': len(historical_draws)
        })
        msg = f"Model trained on {len(historical_draws)} draws and saved to Firestore."
        logging.info(f"--- ML_TRAINING_SUCCESS: {msg} ---")
        return True, msg
    except Exception as e:
        msg = f"An error occurred during model training and saving: {e}"
        logging.error(f"--- ML_TRAINING_ERROR: {msg} ---", exc_info=True)
        return False, msg

def load_model():
    try:
        model_doc = get_ml_model_doc_ref().get()
        if not model_doc.exists:
            logging.warning("ML model not found in Firestore.")
            return None
        base64_encoded_model = model_doc.to_dict()['model_data_base64']
        decoded_model_bytes = base64.b64decode(base64_encoded_model)
        model_file = io.BytesIO(decoded_model_bytes)
        model = joblib.load(model_file)
        logging.info("ML model successfully loaded from Firestore.")
        return model
    except Exception as e:
        logging.error(f"Failed to load ML model from Firestore: {e}", exc_info=True)
        return None

# --- Prediction Strategies ---
def predict_with_ml(base_mains, bonus1, bonus2, historical_draws, target_size=4):
    model = load_model()
    if model is None:
        return None

    logging.info("Using trained ML model for prediction.")
    following_pool = get_following_numbers_pool(base_mains, historical_draws)
    recency_freqs = Counter(num for draw in historical_draws[:15] for num in draw[1])
    
    prediction_features = []
    for num in range(1, 51):
        feature_vector = {
            'is_in_following_pool': 1 if num in following_pool else 0,
            'recency_count_15': recency_freqs.get(num, 0),
            'was_in_prev_mains': 1 if num in base_mains else 0,
            'was_in_prev_bonus': 1 if num in [bonus1, bonus2] else 0,
        }
        prediction_features.append(feature_vector)
    
    df_pred = pd.DataFrame(prediction_features)
    probabilities = model.predict_proba(df_pred)[:, 1]
    prob_series = pd.Series(probabilities, index=range(1, 51))
    
    return prob_series.nlargest(target_size).index.tolist()

def generate_live_prediction(historical_draws):
    if not historical_draws or len(historical_draws) < 20:
        logging.warning("Not enough historical draws to generate a prediction.")
        return None

    latest_draw = historical_draws[0]
    base_mains, bonus1, bonus2 = latest_draw[1], latest_draw[2], latest_draw[3]

    prediction = predict_with_ml(base_mains, bonus1, bonus2, historical_draws[1:])
    strategy_used = 'ml_random_forest_v1'

    if prediction is None:
        logging.warning("ML model not found, prediction failed.")
        return None
    
    return {'strategy_used': strategy_used, 'prediction': prediction}

# --- Data Class for UI ---
class PredictionResult:
    def __init__(self, target_draw_time, strategy_used, bonus, prediction, actual_mains=None, actual_bonuses=None, hits=None):
        self.target_draw_time, self.strategy_used, self.bonus, self.prediction, self.actual_mains, self.actual_bonuses, self.hits = target_draw_time, strategy_used, bonus, prediction, actual_mains, actual_bonuses, hits

def get_next_target_draw_time(current_time_harare):
    now = current_time_harare.astimezone(harare_tz).replace(second=0, microsecond=0)
    today_morning, today_evening = now.replace(hour=8, minute=30), now.replace(hour=20, minute=30)
    if now < today_morning: return today_morning
    if now < today_evening: return today_evening
    return (now + timedelta(days=1)).replace(hour=8, minute=30)

# --- Flask Routes ---
@app.context_processor
def inject_global_data():
    uid_to_check = session.get('view_as_uid', session.get('firebase_uid'))
    is_subscribed = False
    if uid_to_check and db:
        try:
            user_doc = get_user_doc_ref(uid_to_check).get()
            if user_doc.exists: is_subscribed = user_doc.to_dict().get('is_subscribed', False)
        except Exception as e: logging.error(f"Error fetching subscription status: {e}")
    banner_message = "Welcome!"
    if db:
        try:
            banner_doc = get_public_banner_doc_ref().get()
            if banner_doc.exists: banner_message = banner_doc.to_dict().get('message', banner_message)
        except Exception as e: logging.error(f"Error fetching banner message: {e}")
    view_as_user_email = None
    if session.get('view_as_uid'):
        try:
            user_record = auth.get_user(session['view_as_uid'])
            view_as_user_email = user_record.email
        except Exception as e: logging.error(f"Could not fetch email for view_as_uid: {e}")
    return dict(logged_in='firebase_uid' in session, is_subscribed=is_subscribed, is_admin=session.get('is_admin', False), banner_message=banner_message, now=datetime.now(harare_tz), firebase_web_client_config_json=FIREBASE_WEB_CLIENT_CONFIG_JSON, view_as_user_email=view_as_user_email)
    
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id_token = request.json.get('idToken')
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            email = decoded_token.get('email', 'user')
            session['firebase_uid'] = uid
            session['is_admin'] = decoded_token.get('admin', False)
            user_doc_ref = get_user_doc_ref(uid)
            if not user_doc_ref.get().exists:
                user_doc_ref.set({'email': email, 'is_subscribed': False}, merge=True)
            flash(f"Welcome back, {email}!", "success")
            return jsonify(success=True, redirect=url_for('index'))
        except Exception as e: return jsonify(success=False, message=str(e)), 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        id_token = request.json.get('idToken')
        email = request.json.get('email')
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            get_user_doc_ref(uid).set({'email': email, 'is_subscribed': False, 'registered_at': firestore.SERVER_TIMESTAMP})
            users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
            users_count = list(users_ref.limit(2).stream())
            if len(users_count) == 1:
                auth.set_custom_user_claims(uid, {'admin': True})
                session['is_admin'] = True
            session['firebase_uid'] = uid
            flash(f"Welcome, {email}! Your registration was successful.", "success")
            return jsonify(success=True, redirect=url_for('index'))
        except Exception as e: return jsonify(success=False, message=str(e)), 400
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.route('/subscribe_info')
@login_required
def subscribe_info():
    return render_template('subscribe_info.html')

@app.route('/admin_panel')
@login_required
@admin_required
def admin_panel():
    return render_template('admin_panel.html')
    
@app.route('/admin/check_retrain_status')
@login_required
@admin_required
def admin_check_retrain_status():
    try:
        model_doc = get_ml_model_doc_ref().get()
        if not model_doc.exists:
            return jsonify({'status': 'due', 'reason': 'Model has not been trained yet.'})
        last_trained_timestamp = model_doc.to_dict().get('last_trained')
        if not last_trained_timestamp:
            return jsonify({'status': 'due', 'reason': 'Last trained date is missing.'})
        last_trained_date = normalize_to_harare_time(last_trained_timestamp)
        if datetime.now(harare_tz) > last_trained_date + timedelta(days=7):
            return jsonify({'status': 'due', 'reason': 'It has been more than 7 days since the last training.'})
        else:
            days_since_training = (datetime.now(harare_tz) - last_trained_date).days
            return jsonify({'status': 'ok', 'reason': f'Model was trained {days_since_training} day(s) ago.'})
    except Exception as e:
        logging.error(f"Error checking retrain status: {e}")
        return jsonify({'status': 'error', 'reason': str(e)}), 500

@app.route('/force_prediction_update', methods=['POST'])
@login_required
@admin_required
def force_prediction_update():
    scheduler.add_job(scrape_and_process_draws_job, 'date', run_date=datetime.now(harare_tz) + timedelta(seconds=1), id='manual_scrape_job', replace_existing=True)
    flash("Prediction update initiated successfully!", "success")
    return redirect(url_for('admin_panel'))

@app.route('/admin/train_model', methods=['POST'])
@login_required
@admin_required
def admin_train_model():
    success, message = train_and_save_model()
    flash(message, "success" if success else "error")
    return redirect(url_for('admin_panel'))
    
@app.route('/admin/start_backtest', methods=['POST'])
@login_required
@admin_required
def admin_start_backtest():
    scheduler.add_job(backtest_strategy_job, 'date', run_date=datetime.now(harare_tz) + timedelta(seconds=1), id='manual_backtest_job', replace_existing=True)
    flash("ML backtest started in the background. Results will be available on the backtest page shortly.", "info")
    return redirect(url_for('admin_panel'))

@app.route('/backtest_results')
@login_required
@admin_required
def backtest_results_page():
    backtest_data_doc = get_backtest_results_doc_ref().get()
    if not backtest_data_doc.exists:
        return render_template('backtest_results.html', backtest_status='not_run')
    
    backtest_data = backtest_data_doc.to_dict()
    status = backtest_data.get('status', 'unknown')
    
    # Pass all data to template regardless of status, template will handle display logic
    for result in backtest_data.get('results', []):
        if 'target_draw_time' in result and isinstance(result['target_draw_time'], str):
            result['target_draw_time'] = datetime.fromisoformat(result['target_draw_time'])

    return render_template('backtest_results.html', backtest_status=status, **backtest_data)


@app.route('/telegram_settings', methods=['GET', 'POST'])
@login_required
@subscription_required
def telegram_settings():
    user_doc_ref = get_user_doc_ref(session['firebase_uid'])
    if request.method == 'POST':
        chat_id = request.form.get('telegram_chat_id', '').strip()
        if chat_id:
            user_doc_ref.update({'telegram_chat_id': chat_id})
            flash("Telegram Chat ID updated!", "success")
        return redirect(url_for('telegram_settings'))
    user_data = user_doc_ref.get().to_dict() or {}
    return render_template('telegram_settings.html', current_chat_id=user_data.get('telegram_chat_id', ''))

@app.route('/send_alert', methods=['POST'])
@login_required
@subscription_required
def send_alert():
    user_doc = get_user_doc_ref(session['firebase_uid']).get()
    chat_id = user_doc.to_dict().get('telegram_chat_id') if user_doc.exists else None
    if not chat_id:
        flash("Telegram Chat ID not set.", "error")
        return redirect(url_for('telegram_settings'))
    
    pred_doc = get_public_prediction_doc_ref().get()
    if not pred_doc.exists:
        flash("No current prediction available.", "error")
        return redirect(url_for('index'))
        
    pred_data = pred_doc.to_dict()
    msg = f"🔮 Hilzhosting 5/50 Prediction 🔮\n\nMains: <b>{', '.join(map(str, pred_data.get('prediction', [])))}</b>\nBonuses: <b>{', '.join(map(str, pred_data.get('bonus', [])))}</b>\n\nGood luck!"
    success, message = send_telegram_message(msg, chat_id)
    flash(message, "success" if success else "error")
    return redirect(url_for('index'))



# --- Backtest Background Job with Positional Analysis ---
def backtest_strategy_job():
    with app.app_context():
        logging.info("--- 🚀 STARTING ML BACKTEST WITH POSITIONAL ANALYSIS 🚀 ---")
        
        try:
            draws_data = get_historical_draws_from_firestore(history_window_days=365)
            if len(draws_data) < 50:
                reason = f"Not enough draws for backtest. Need at least 50, have {len(draws_data)}."
                get_backtest_results_doc_ref().set({'status': 'failed', 'reason': reason})
                return

            get_backtest_results_doc_ref().set({'status': 'running', 'last_started': firestore.SERVER_TIMESTAMP, 'progress': '0%'})

            results = []
            positional_hit_counts = [0] * 4 # [P1 hits, P2 hits, P3 hits, P4 hits]
            total_tests = len(draws_data) - 40
            
            for i in range(total_tests):
                target_draw = draws_data[i]
                base_draw = draws_data[i+1]
                history_for_pred = draws_data[i+2:]
                
                # Use a consistent temporary model for the entire backtest
                X_train, y_train = create_feature_dataset(history_for_pred)
                if X_train is None: continue
                
                temp_model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
                temp_model.fit(X_train, y_train)

                # Generate features for prediction
                base_mains, b1, b2 = base_draw[1], base_draw[2], base_draw[3]
                following_pool = get_following_numbers_pool(base_mains, history_for_pred)
                recency_freqs = Counter(num for draw in history_for_pred[:15] for num in draw[1])
                
                pred_features = []
                for num in range(1, 51):
                    pred_features.append({
                        'is_in_following_pool': 1 if num in following_pool else 0,
                        'recency_count_15': recency_freqs.get(num, 0),
                        'was_in_prev_mains': 1 if num in base_mains else 0,
                        'was_in_prev_bonus': 1 if num in [b1, b2] else 0,
                    })
                
                df_pred = pd.DataFrame(pred_features)
                probabilities = temp_model.predict_proba(df_pred)[:, 1]
                prob_series = pd.Series(probabilities, index=range(1, 51))
                predicted_mains = prob_series.nlargest(4).index.tolist()

                # Evaluate hits
                target_mains_set = set(target_draw[1])
                hits = len(set(predicted_mains).intersection(target_mains_set))
                
                positional_hits = []
                for j, p_num in enumerate(predicted_mains):
                    is_hit = p_num in target_mains_set
                    positional_hits.append(is_hit)
                    if is_hit: positional_hit_counts[j] += 1

                results.append({
                    'target_draw_time': target_draw[0].isoformat(),
                    'prediction': predicted_mains,
                    'actual_mains': target_draw[1],
                    'hits': hits,
                    'positional_hits': positional_hits
                })
                
                progress = int(((i + 1) / total_tests) * 100) if total_tests > 0 else 0
                get_backtest_results_doc_ref().update({'progress': f'{progress}%'})

            total_hits = sum(r['hits'] for r in results)
            total_predictions = len(results)
            avg_hit_rate = (total_hits / (total_predictions * 4)) * 100 if total_predictions > 0 else 0

            payload = {
                'status': 'completed',
                'last_completed': firestore.SERVER_TIMESTAMP,
                'results': results,
                'total_backtests': total_predictions,
                'average_hit_rate': avg_hit_rate,
                'total_3_hits': sum(1 for r in results if r['hits'] >= 3),
                'total_2_hits': sum(1 for r in results if r['hits'] >= 2),
                'positional_hit_counts': positional_hit_counts,
            }
            get_backtest_results_doc_ref().set(payload)
            logging.info(f"--- ✅ ML BACKTEST FINISHED ✅ ---")
        except Exception as e:
            logging.error(f"--- ❌ ML Backtest Failed: {e} ---", exc_info=True)
            get_backtest_results_doc_ref().set({'status': 'failed', 'reason': str(e)})


# --- Scheduler and Jobs ---
scheduler = BackgroundScheduler(daemon=True, timezone=harare_tz)
def scrape_and_process_draws_job():
    logging.info("--- JOB START: Scrape and Process Draws ---")
    draws_data, error = fetch_draws_from_website()
    if error:
        logging.error(f"Scraping job failed: {error}")
        return
    store_draws_to_firestore(draws_data)

    historical_draws = get_historical_draws_from_firestore(history_window_days=90)
    live_prediction_result = generate_live_prediction(historical_draws)

    if live_prediction_result:
        prediction_payload = {
            'target_draw_time': get_next_target_draw_time(datetime.now(harare_tz)),
            'strategy_used': live_prediction_result['strategy_used'],
            'prediction': live_prediction_result['prediction'],
            'bonus': [historical_draws[0][2], historical_draws[0][3]],
            'timestamp_generated': firestore.SERVER_TIMESTAMP
        }
        get_public_prediction_doc_ref().set(prediction_payload, merge=True)
        logging.info(f"Saved new public prediction: {live_prediction_result['prediction']}")
    else:
        logging.error("Failed to generate live prediction.")

scheduler.add_job(scrape_and_process_draws_job, 'cron', hour='8,20', minute='55', id='scrape_process_job', replace_existing=True)

if __name__ == '__main__':
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    # Use a proper WSGI server in production
    # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
