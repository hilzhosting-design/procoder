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
    """Initializes the Firebase Admin SDK and GCS client using the same credentials."""
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

def get_strategy_insights_doc_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_settings').document('strategy_insights')

def get_backtest_results_doc_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_settings').document('backtest_results')

def get_user_doc_ref(firebase_uid):
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('users').document(firebase_uid)

def get_user_predictions_history_ref(firebase_uid):
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('users').document(firebase_uid).collection('predictions_history')

def get_draws_collection_ref():
    return db.collection('artifacts').document(get_app_id_for_firestore()).collection('public_draw_results')

# --- Telegram Integration ---
def send_telegram_message(message, chat_id):
    if not TELEGRAM_API_URL or not chat_id: return False, "Telegram not configured."
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(TELEGRAM_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return True, "Message sent successfully!"
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return False, f"Failed to send Telegram message: {e}"

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
    return [], "Unexpected error."

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
            draws.append((timestamp_dt, sorted(mains), bonuses[0], bonuses[1], data.get('draw_type', 'Unknown')))
    except Exception as e: logging.error(f"Failed to fetch historical draws: {e}")
    return draws

def get_latest_actual_draw():
    draws = get_historical_draws_from_firestore(limit=1)
    if draws: return draws[0]
    return None, None, None, None, None

# --- ML Feature Helpers ---
def super_hybrid_pool(bonus):
    mirror = 50 - bonus
    pool = list(set([bonus, mirror, bonus + 1, bonus - 1, bonus + 10, bonus - 10]))
    return [n for n in pool if 1 <= n <= 50]

def get_hot_numbers(historical_draws, window=15, count=5):
    if not historical_draws or len(historical_draws) < window: return []
    all_numbers = [num for _, mains, _, _, _ in historical_draws[:window] for num in mains]
    return [num for num, _ in Counter(all_numbers).most_common(count)]

def get_overdue_numbers(historical_draws, count=5):
    if not historical_draws: return []
    last_seen = {num: i + 1 for i, draw in enumerate(reversed(historical_draws)) for num in draw[1]}
    if not last_seen: return []
    draws_since_seen = {num: len(historical_draws) - (pos - 1) for num, pos in last_seen.items()}
    overdue = sorted(draws_since_seen.items(), key=lambda item: item[1], reverse=True)
    return [num for num, _ in overdue[:count]]

def get_strongest_pairs(historical_draws, window=100, count=5):
    if not historical_draws: return []
    pair_counts = Counter(pair for _, mains, _, _, _ in historical_draws[:window] for pair in itertools.combinations(sorted(mains), 2))
    strong_numbers = {num for pair, _ in pair_counts.most_common(count) for num in pair}
    return list(strong_numbers)

# --- Machine Learning Section with Firestore ---
def create_feature_dataset(historical_draws):
    features = []
    if len(historical_draws) < 25: return None, None
    for i in range(len(historical_draws) - 20):
        target_draw_mains = set(historical_draws[i][1])
        base_draw = historical_draws[i+1]
        prediction_history = historical_draws[i+2:]
        _, base_mains, b1, b2, _ = base_draw
        hot_pool = get_hot_numbers(prediction_history)
        overdue_pool = get_overdue_numbers(prediction_history)
        pair_pool = get_strongest_pairs(prediction_history)
        bonus_pool = set(super_hybrid_pool(b1) + super_hybrid_pool(b2))
        for num in range(1, 51):
            features.append({'number': num, 'is_hot': 1 if num in hot_pool else 0, 'is_overdue': 1 if num in overdue_pool else 0, 'is_in_pair': 1 if num in pair_pool else 0, 'is_in_bonus_pool': 1 if num in bonus_pool else 0, 'was_in_prev_mains': 1 if num in base_mains else 0, 'was_in_prev_bonus': 1 if num in [b1, b2] else 0, 'was_winner': 1 if num in target_draw_mains else 0})
    df = pd.DataFrame(features)
    return df.drop(['was_winner', 'number'], axis=1), df['was_winner']

def train_and_save_model():
    logging.info("--- ML_TRAINING: Starting model training process. ---")
    try:
        historical_draws = get_historical_draws_from_firestore()
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

        model_doc_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('ml_models').document('prediction_model')
        model_doc_ref.set({
            'model_data_base64': base64_encoded_model,
            'last_trained': firestore.SERVER_TIMESTAMP
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
        model_doc_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('ml_models').document('prediction_model')
        model_doc = model_doc_ref.get()
        if not model_doc.exists:
            logging.warning("Model document not found in Firestore.")
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
def predict_strategy_v11_fallback(base_mains, bonus1, bonus2, historical_draws):
    scores = Counter()
    calculated_auto_entry = (bonus1 + bonus2) * 2
    auto_entry_number = calculated_auto_entry if 1 <= calculated_auto_entry <= 50 else (bonus1 + bonus2 if 1 <= (bonus1 + bonus2) <= 50 else bonus1)
    sum_mains = sum(base_mains)
    decimal_bonus = float(f"{bonus1}.{bonus2}")
    code = int(sum_mains + decimal_bonus)
    code_str = str(code)
    lotto_strategy_number = bonus2
    if len(code_str) >= 2:
        calculated_number = int(code_str[:-1]) - int(code_str[-1])
        if 1 <= calculated_number <= 50: lotto_strategy_number = calculated_number
    for num in {auto_entry_number, lotto_strategy_number}: scores[num] += 3
    for num in set(get_hot_numbers(historical_draws) + get_overdue_numbers(historical_draws)): scores[num] += 1
    for num in get_strongest_pairs(historical_draws): scores[num] += 2
    for num in set(super_hybrid_pool(bonus1) + super_hybrid_pool(bonus2)): scores[num] += 1
    if not scores: return get_hot_numbers(historical_draws, window=20, count=4)
    return sorted([num for num, _ in scores.most_common(4)])

def predict_strategy(base_mains, bonus1, bonus2, historical_draws, target_size=4):
    model = load_model()
    if model is None:
        logging.warning("ML model not available. Using v11 fallback strategy.")
        return predict_strategy_v11_fallback(base_mains, bonus1, bonus2, historical_draws)
    logging.info("Using trained ML model from Firestore for prediction.")
    hot_pool, overdue_pool, pair_pool = get_hot_numbers(historical_draws), get_overdue_numbers(historical_draws), get_strongest_pairs(historical_draws)
    bonus_pool = set(super_hybrid_pool(bonus1) + super_hybrid_pool(bonus2))
    prediction_features = [{'is_hot': 1 if num in hot_pool else 0, 'is_overdue': 1 if num in overdue_pool else 0, 'is_in_pair': 1 if num in pair_pool else 0, 'is_in_bonus_pool': 1 if num in bonus_pool else 0, 'was_in_prev_mains': 1 if num in base_mains else 0, 'was_in_prev_bonus': 1 if num in [bonus1, bonus2] else 0} for num in range(1, 51)]
    df_pred = pd.DataFrame(prediction_features)
    probabilities = model.predict_proba(df_pred)[:, 1]
    prob_series = pd.Series(probabilities, index=range(1, 51))
    return sorted(prob_series.nlargest(target_size).index.tolist())

def generate_live_prediction(historical_draws):
    if not historical_draws or len(historical_draws) < 3: return None
    latest_draw = historical_draws[0]
    base_mains, bonus1, bonus2 = latest_draw[1], latest_draw[2], latest_draw[3]
    prediction = predict_strategy(base_mains, bonus1, bonus2, historical_draws[1:])
    
    model_doc = db.collection('artifacts').document(get_app_id_for_firestore()).collection('ml_models').document('prediction_model').get()
    strategy_name = "ml_random_forest_firestore" if model_doc.exists else "dynamic_scoring_v11_fallback"
    
    return {'strategy_used': strategy_name, 'prediction': prediction}

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

@app.route('/')
def index():
    if db is None:
        flash("Database not initialized.", "error")
        return render_template('index.html', error="Database connection error.")
    uid_to_view = session.get('view_as_uid', session.get('firebase_uid'))
    latest_draw_time, latest_mains, b1, b2, _ = get_latest_actual_draw()
    latest_bonuses = [b1, b2] if b1 else []
    live_prediction_history = []
    is_subscribed_for_view = False
    if uid_to_view:
        try:
            user_doc = get_user_doc_ref(uid_to_view).get()
            if user_doc.exists: is_subscribed_for_view = user_doc.to_dict().get('is_subscribed', False)
        except Exception as e: logging.error(f"Error checking subscription status for {uid_to_view}: {e}")
    history_ref = get_user_predictions_history_ref(uid_to_view) if uid_to_view and is_subscribed_for_view else get_public_prediction_history_ref()
    try:
        history_docs = history_ref.order_by('target_draw_time', direction=firestore.Query.DESCENDING).limit(10).stream()
        for doc in history_docs:
            data = doc.to_dict()
            live_prediction_history.append(PredictionResult(normalize_to_harare_time(data.get('target_draw_time')), data.get('strategy_used'), data.get('bonus'), data.get('prediction'), data.get('actual_mains'), data.get('actual_bonuses'), data.get('hits')))
    except Exception as e: logging.error(f"Could not fetch prediction history: {e}")
    current_prediction_vs_actual = live_prediction_history[0] if live_prediction_history else None
    combos_3, combos_2 = [], []
    if current_prediction_vs_actual and current_prediction_vs_actual.prediction:
        combos_3 = list(itertools.combinations(current_prediction_vs_actual.prediction, 3))
        combos_2 = list(itertools.combinations(current_prediction_vs_actual.prediction, 2))
    return render_template('index.html', latest_draw_time=latest_draw_time, latest_actual_mains=latest_mains, latest_actual_bonuses=latest_bonuses, current_prediction_vs_actual=current_prediction_vs_actual, generated_main_combos_3=combos_3, generated_main_combos_2=combos_2, live_prediction_history=live_prediction_history, is_subscribed=is_subscribed_for_view)

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

@app.route('/telegram_settings', methods=['GET', 'POST'])
@login_required
@subscription_required
def telegram_settings():
    uid_to_view = session.get('view_as_uid', session.get('firebase_uid'))
    user_doc_ref = get_user_doc_ref(uid_to_view)
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
    uid_to_view = session.get('view_as_uid', session.get('firebase_uid'))
    user_doc = get_user_doc_ref(uid_to_view).get()
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

@app.route('/admin_panel')
@login_required
@admin_required
def admin_panel():
    return render_template('admin_panel.html')

@app.route('/force_prediction_update', methods=['POST'])
@login_required
@admin_required
def force_prediction_update():
    try:
        scrape_and_process_draws_job()
        flash("Prediction update initiated successfully!", "success")
    except Exception as e:
        flash(f"Error forcing prediction update: {e}", "error")
    return redirect(url_for('admin_panel'))

@app.route('/admin/train_model', methods=['POST'])
@login_required
@admin_required
def admin_train_model():
    try:
        success, message = train_and_save_model()
        flash(message, "success" if success else "error")
    except Exception as e:
        flash(f"A critical error occurred while triggering model training: {e}", "error")
        logging.error(f"Critical error in admin_train_model route: {e}")
    return redirect(url_for('admin_panel'))

@app.route('/admin_banner_settings', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_banner_settings():
    banner_doc_ref = get_public_banner_doc_ref()
    if request.method == 'POST':
        new_message = request.form.get('banner_message')
        if new_message:
            try:
                banner_doc_ref.set({'message': new_message})
                flash("Banner message updated successfully!", "success")
            except Exception as e: flash(f"Error updating banner: {e}", "error")
        return redirect(url_for('admin_banner_settings'))
    current_message = "Welcome!"
    try:
        banner_doc = banner_doc_ref.get()
        if banner_doc.exists: current_message = banner_doc.to_dict().get('message', current_message)
    except Exception as e: flash(f"Error fetching current banner message: {e}", "error")
    return render_template('admin_banner_settings.html', current_message=current_message)

@app.route('/admin_users')
@login_required
@admin_required
def admin_users():
    users_list = []
    try:
        users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
        for doc in users_ref.stream():
            user_data = doc.to_dict()
            uid = doc.id
            is_admin = False
            try:
                user_record = auth.get_user(uid)
                if user_record.custom_claims and user_record.custom_claims.get('admin'):
                    is_admin = True
            except Exception as e: logging.warning(f"Could not fetch claims for user {uid}: {e}")
            users_list.append({**user_data, 'uid': uid, 'is_admin': is_admin})
    except Exception as e: flash(f"Error fetching users: {e}", "error")
    return render_template('admin_users.html', users=users_list)

@app.route('/set_admin_claim', methods=['GET', 'POST'])
@login_required
@admin_required
def set_admin_claim():
    if request.method == 'POST':
        email = request.form.get('email')
        action = request.form.get('action')
        try:
            user = auth.get_user_by_email(email)
            if action == 'grant':
                auth.set_custom_user_claims(user.uid, {'admin': True})
                flash(f"Admin privileges granted to {email}.", "success")
            elif action == 'revoke':
                auth.set_custom_user_claims(user.uid, {'admin': False})
                flash(f"Admin privileges revoked for {email}.", "success")
        except Exception as e:
            flash(f"Error updating claims for {email}: {e}", "error")
        return redirect(url_for('admin_users'))
    return render_template('admin_set_admin_claim.html')

@app.route('/admin_toggle_subscription/<uid>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_subscription(uid):
    try:
        user_doc_ref = get_user_doc_ref(uid)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            new_status = not user_doc.to_dict().get('is_subscribed', False)
            user_doc_ref.update({'is_subscribed': new_status})
            flash(f"Subscription for user set to {new_status}.", "success")
    except Exception as e: flash(f"Error toggling subscription: {e}", "error")
    return redirect(url_for('admin_users'))

@app.route('/admin_delete_user/<uid>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(uid):
    try:
        auth.delete_user(uid)
        get_user_doc_ref(uid).delete()
        flash(f"User {uid} deleted successfully.", "success")
    except Exception as e:
        flash(f"Error deleting user {uid}: {e}", "error")
    return redirect(url_for('admin_users'))

@app.route('/admin_send_results', methods=['POST'])
@login_required
@admin_required
def admin_send_results():
    try:
        send_results_and_hits_job()
        flash("Results and hits notifications sent successfully!", "success")
    except Exception as e:
        flash(f"Error sending results notifications: {e}", "error")
        logging.error(f"Error manually sending results notifications: {e}")
    return redirect(url_for('admin_panel'))

@app.route('/admin/view_as/<uid>')
@login_required
@admin_required
def admin_view_as(uid):
    session['view_as_uid'] = uid
    return redirect(url_for('index'))

@app.route('/admin/exit_view_as')
@login_required
@admin_required
def admin_exit_view_as():
    session.pop('view_as_uid', None)
    return redirect(url_for('admin_users'))
    
# --- Backtest Background Job ---
def backtest_strategy_job():
    """Performs a true walk-forward validation backtest for the ML model as a background job."""
    with app.app_context():
        logging.info("--- BACKGROUND JOB: Starting ML Walk-Forward Backtest ---")
        try:
            draws_data = get_historical_draws_from_firestore()
            results = []
            
            initial_training_size = 50
            if len(draws_data) < initial_training_size + 10:
                logging.warning(f"Not enough draws for a valid ML backtest. Need {initial_training_size + 10}, have {len(draws_data)}.")
                get_backtest_results_doc_ref().set({'status': 'failed', 'reason': 'Not enough data'})
                return

            get_backtest_results_doc_ref().set({'status': 'running', 'last_started': firestore.SERVER_TIMESTAMP, 'progress': '0%'})

            walk_forward_draws = list(reversed(draws_data))
            total_predictions = len(walk_forward_draws) - initial_training_size

            for i in range(initial_training_size, len(walk_forward_draws)):
                training_history = walk_forward_draws[:i]
                base_draw_for_prediction = training_history[-1]
                target_actual_draw = walk_forward_draws[i]
                
                X_train, y_train = create_feature_dataset(list(reversed(training_history)))
                if X_train is None: continue
                    
                temp_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                temp_model.fit(X_train, y_train)
                
                base_mains, b1, b2 = base_draw_for_prediction[1], base_draw_for_prediction[2], base_draw_for_prediction[3]
                history_for_features = list(reversed(training_history[:-1]))
                
                hot_pool, overdue_pool, pair_pool = get_hot_numbers(history_for_features), get_overdue_numbers(history_for_features), get_strongest_pairs(history_for_features)
                bonus_pool = set(super_hybrid_pool(b1) + super_hybrid_pool(b2))
                
                prediction_features = [{'is_hot': 1 if num in hot_pool else 0, 'is_overdue': 1 if num in overdue_pool else 0, 'is_in_pair': 1 if num in pair_pool else 0, 'is_in_bonus_pool': 1 if num in bonus_pool else 0, 'was_in_prev_mains': 1 if num in base_mains else 0, 'was_in_prev_bonus': 1 if num in [b1, b2] else 0} for num in range(1, 51)]
                df_pred = pd.DataFrame(prediction_features)
                
                probabilities = temp_model.predict_proba(df_pred)[:, 1]
                prob_series = pd.Series(probabilities, index=range(1, 51))
                predicted_mains = sorted(prob_series.nlargest(4).index.tolist())
                
                target_timestamp, target_mains, target_b1, target_b2, target_draw_type = target_actual_draw
                hits = len(set(predicted_mains).intersection(set(target_mains)))
                
                results.append({'target_draw_time': target_timestamp.isoformat(), 'draw_date': target_timestamp.strftime('%Y-%m-%d'), 'draw_type': target_draw_type, 'strategy_used': 'ml_walk_forward', 'bonus': [b1, b2], 'prediction': predicted_mains, 'actual_mains': target_mains, 'actual_bonuses': [target_b1, target_b2], 'hits': hits})
                
                progress = int(((i - initial_training_size + 1) / total_predictions) * 100)
                if progress % 10 == 0:
                    get_backtest_results_doc_ref().update({'progress': f'{progress}%'})

            results.reverse()
            main_ranking = Counter(num for _, mains, _, _, _ in draws_data for num in mains).most_common()
            bonus_ranking = Counter(b for _, _, b1, b2, _ in draws_data for b in (b1, b2)).most_common()
            
            payload = {
                'status': 'completed',
                'last_completed': firestore.SERVER_TIMESTAMP,
                'results': results,
                'main_ranking': main_ranking,
                'bonus_ranking': bonus_ranking
            }
            get_backtest_results_doc_ref().set(payload)
            logging.info(f"--- BACKGROUND JOB: ML Walk-Forward Backtest Finished. Processed {len(results)} predictions. ---")
        except Exception as e:
            logging.error(f"--- BACKGROUND JOB: ML Backtest Failed: {e} ---", exc_info=True)
            get_backtest_results_doc_ref().set({'status': 'failed', 'reason': str(e)})

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
    if status == 'completed':
        backtest_results = backtest_data.get('results', [])
        for result in backtest_results:
            if 'target_draw_time' in result and isinstance(result['target_draw_time'], str):
                result['target_draw_time'] = datetime.fromisoformat(result['target_draw_time'])
        main_ranking = backtest_data.get('main_ranking', [])
        bonus_ranking = backtest_data.get('bonus_ranking', [])
        total_hits = sum(r['hits'] for r in backtest_results)
        total_predicted = sum(len(r.get('prediction', [])) for r in backtest_results)
        avg_hit_rate = (total_hits / total_predicted) * 100 if total_predicted > 0 else 0
        return render_template('backtest_results.html', backtest_status=status, last_completed=normalize_to_harare_time(backtest_data.get('last_completed')), backtest_results=backtest_results, main_ranking=main_ranking, bonus_ranking=bonus_ranking, total_backtests=len(backtest_results), average_hit_rate=avg_hit_rate, total_3_hits=sum(1 for r in backtest_results if r['hits'] >= 3), total_2_hits=sum(1 for r in backtest_results if r['hits'] >= 2))
    elif status == 'running':
        last_started = normalize_to_harare_time(backtest_data.get('last_started'))
        progress = backtest_data.get('progress', '0%')
        return render_template('backtest_results.html', backtest_status=status, last_started=last_started, progress=progress)
    else:
        return render_template('backtest_results.html', backtest_status=status, reason=backtest_data.get('reason'))

# --- Scheduler and Jobs ---
scheduler = BackgroundScheduler(daemon=True, timezone=harare_tz)
HISTORY_WINDOW_DAYS = 90

def scrape_and_process_draws_job():
    logging.info("--- JOB START: Scrape and Process Draws ---")
    draws_data, error = fetch_draws_from_website()
    if error:
        logging.error(f"Scraping job failed: {error}")
        return
    store_draws_to_firestore(draws_data)
    historical_draws = get_historical_draws_from_firestore(history_window_days=HISTORY_WINDOW_DAYS)
    live_prediction_result = generate_live_prediction(historical_draws)
    if live_prediction_result:
        prediction_payload = {'target_draw_time': get_next_target_draw_time(datetime.now(harare_tz)), 'strategy_used': live_prediction_result['strategy_used'], 'bonus': [historical_draws[0][2], historical_draws[0][3]], 'prediction': live_prediction_result['prediction'], 'actual_mains': [], 'actual_bonuses': [], 'hits': None, 'timestamp_generated': firestore.SERVER_TIMESTAMP}
        try:
            prediction_doc_id = prediction_payload['target_draw_time'].strftime('%Y-%m-%d_%H%M')
            get_public_prediction_history_ref().document(prediction_doc_id).set(prediction_payload)
            get_public_prediction_doc_ref().set(prediction_payload, merge=True)
            logging.info(f"Saved new prediction for {prediction_doc_id}.")
        except Exception as e: logging.error(f"Failed to save public prediction: {e}")
    else:
        logging.error("Failed to generate live prediction.")
        return
    update_all_user_predictions_job()
    check_and_update_prediction_hits_job()
    logging.info("--- JOB END: Scrape and Process Draws ---")

def update_all_user_predictions_job():
    public_pred_doc = get_public_prediction_doc_ref().get()
    if not public_pred_doc.exists: return
    pred_data = public_pred_doc.to_dict()
    target_time = normalize_to_harare_time(pred_data.get('target_draw_time'))
    pred_doc_id = target_time.strftime('%Y-%m-%d_%H%M')
    users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
    for user in users_ref.where(filter=FieldFilter('is_subscribed', '==', True)).stream():
        history_ref = get_user_predictions_history_ref(user.id)
        if not history_ref.document(pred_doc_id).get().exists:
            history_ref.document(pred_doc_id).set(pred_data, merge=True)
            logging.info(f"Saved prediction for user {user.id}")

def check_and_update_prediction_hits_job():
    latest_draw_time, latest_mains, b1, b2, _ = get_latest_actual_draw()
    if not latest_draw_time: return
    target_time = latest_draw_time.replace(second=0, microsecond=0)
    target_time = target_time.replace(hour=8, minute=30) if latest_draw_time.hour < 12 else target_time.replace(hour=20, minute=30)
    pred_doc_id = target_time.strftime('%Y-%m-%d_%H%M')
    update_payload = {'actual_mains': latest_mains, 'actual_bonuses': [b1, b2]}
    public_pred_ref = get_public_prediction_history_ref().document(pred_doc_id)
    public_pred_doc = public_pred_ref.get()
    if public_pred_doc.exists and not public_pred_doc.to_dict().get('actual_mains'):
        hits = len(set(public_pred_doc.to_dict().get('prediction', [])).intersection(set(latest_mains)))
        public_pred_ref.update({**update_payload, 'hits': hits})
    users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
    for user_doc in users_ref.stream():
        pred_ref = get_user_predictions_history_ref(user_doc.id).document(pred_doc_id)
        prediction = pred_ref.get()
        if prediction.exists and not prediction.to_dict().get('actual_mains'):
            hits = len(set(prediction.to_dict().get('prediction', [])).intersection(set(latest_mains)))
            pred_ref.update({**update_payload, 'hits': hits})

def precompute_successful_bonuses_job():
    logging.info("--- JOB START: Pre-compute Successful Bonuses ---")
    # This job now depends on the results of the backtest job.
    # We will read from the saved backtest results instead of running it live.
    backtest_doc = get_backtest_results_doc_ref().get()
    if not backtest_doc.exists or backtest_doc.to_dict().get('status') != 'completed':
        logging.warning("Skipping precompute bonuses job: No completed backtest found.")
        return
    backtest_results = backtest_doc.to_dict().get('results', [])
    successful_bonuses = set()
    for result in backtest_results:
        if result.get('hits', 0) >= 3:
            bonuses = result.get('bonus', [])
            for bonus_num in bonuses: successful_bonuses.add(bonus_num)
    if successful_bonuses:
        get_strategy_insights_doc_ref().set({'bonuses_with_3_hits': sorted(list(successful_bonuses)), 'last_updated': firestore.SERVER_TIMESTAMP})
        logging.info(f"Stored {len(successful_bonuses)} successful bonuses.")
    logging.info("--- JOB END: Pre-compute Successful Bonuses ---")

def get_subscribed_users_with_telegram():
    if db is None: return []
    try:
        users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
        query = users_ref.where(filter=FieldFilter('is_subscribed', '==', True))
        return [{'uid': doc.id, 'chat_id': doc.to_dict().get('telegram_chat_id')} for doc in query.stream() if doc.to_dict().get('telegram_chat_id')]
    except Exception as e:
        logging.error(f"Failed to get subscribed users with telegram: {e}")
        return []

def send_prediction_alerts_job():
    logging.info("--- JOB: Sending prediction alerts ---")
    users = get_subscribed_users_with_telegram()
    if not users: return
    pred_doc = get_public_prediction_doc_ref().get()
    if not pred_doc.exists: return
    pred_data = pred_doc.to_dict()
    target_time = normalize_to_harare_time(pred_data.get('target_draw_time'))
    if not target_time or (target_time - datetime.now(harare_tz)).total_seconds() > 3 * 3600: return
    msg = f"🔮 *Upcoming 5/50 Prediction* 🔮\n\nFor the *{target_time.strftime('%H:%M')}* draw:\n\nMains: <b>{', '.join(map(str, pred_data.get('prediction', [])))}</b>\nBonuses: <b>{', '.join(map(str, pred_data.get('bonus', [])))}</b>\n\n✨Good luck! ✨"
    for user in users:
        send_telegram_message(msg, user['chat_id'])
        time.sleep(0.5)

def send_results_and_hits_job():
    logging.info("--- JOB: Sending results and hits ---")
    latest_draw_time, latest_mains, b1, b2, _ = get_latest_actual_draw()
    if not latest_draw_time or not latest_mains: return
    users = get_subscribed_users_with_telegram()
    if not users: return
    target_time = latest_draw_time.replace(second=0, microsecond=0)
    target_time = target_time.replace(hour=8, minute=30) if target_time.hour < 12 else target_time.replace(hour=20, minute=30)
    pred_doc_id = target_time.strftime('%Y-%m-%d_%H%M')
    for user in users:
        try:
            pred_doc = get_user_predictions_history_ref(user['uid']).document(pred_doc_id).get()
            if not pred_doc.exists: continue
            pred_data = pred_doc.to_dict()
            user_pred = pred_data.get('prediction', [])
            hits = set(user_pred).intersection(set(latest_mains))
            hits_str = ", ".join(map(str, sorted(list(hits)))) if hits else "None"
            hit_count = len(hits)
            msg = f"🎉 *Draw Results & Your Hits* 🎉\n\nResults for *{latest_draw_time.strftime('%Y-%m-%d %H:%M')}*:\nMains: <b>{', '.join(map(str, latest_mains))}</b>\nBonuses: <b>{b1}, {b2}</b>\n\nYour Prediction: {', '.join(map(str, user_pred))}\nYour Hits: ✅ <b>{hits_str}</b>\n\n"
            if hit_count >= 3: msg += f"🎯 *Congratulations! You got {hit_count} hits!* 🎯"
            elif hit_count > 0: msg += f"Good job on {hit_count} hit(s)!"
            else: msg += "Better luck next time! 🤞"
            send_telegram_message(msg, user['chat_id'])
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Failed to process results for {user['uid']}: {e}")

# Schedule All Jobs
scheduler = BackgroundScheduler(daemon=True, timezone=harare_tz)
scheduler.add_job(scrape_and_process_draws_job, 'cron', hour='8,20', minute='59', id='scrape_process_job', replace_existing=True)
scheduler.add_job(precompute_successful_bonuses_job, 'cron', hour='3', minute='0', id='precompute_bonuses_job', replace_existing=True)
scheduler.add_job(send_prediction_alerts_job, 'cron', hour='7,19', minute='30', id='send_prediction_alerts_job', replace_existing=True)
scheduler.add_job(send_results_and_hits_job, 'cron', hour='8,20', minute='35,40', id='send_results_hits_job', replace_existing=True)

scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
