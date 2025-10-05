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
import time # Import the time module for sleep

# Set logging level to DEBUG to capture all detailed messages
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')

from apscheduler.schedulers.background import BackgroundScheduler

import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth_module
from flask import Flask, render_template, request, flash, redirect, url_for, session, has_request_context, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1.base_client import DocumentSnapshot # Import for type checking Firestore Timestamps


app = Flask(__name__)
print(f"[{datetime.now().isoformat()}] DEBUG: Flask app instance 'app' has been created.")
app.secret_key = os.urandom(24)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" if TELEGRAM_BOT_TOKEN else None

INITIAL_ADMIN_EMAIL = os.getenv("INITIAL_ADMIN_EMAIL", "initial_admin@example.com")

db = None
auth = None
harare_tz = pytz.timezone('Africa/Harare')
utc_tz = pytz.utc

def initialize_firebase_admin_sdk():
    """Initializes the Firebase Admin SDK using credentials from environment variables."""
    global db, auth
    if not firebase_admin._apps:
        try:
            firebase_config_str = os.getenv('__firebase_config')
            if firebase_config_str:
                firebase_config = json.loads(firebase_config_str)
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                logging.info("Firebase Admin SDK initialized using __firebase_config.")
            else:
                logging.warning("__firebase_config not found. Firebase Admin SDK might not initialize correctly locally without a service account key.")
                return False
        except Exception as e:
            logging.error(f"Failed to initialize Firebase Admin SDK: {e}")
            return False

    if firebase_admin._apps:
        db = firestore.client()
        auth = firebase_auth_module
        logging.info(f"Firebase DB and Auth clients obtained. DB is {'initialized' if db else 'NOT initialized'}, Auth is {'initialized' if auth else 'NOT initialized'}.")
        return True
    return False

# Attempt Firebase initialization on app startup
if not initialize_firebase_admin_sdk():
    logging.critical("Firebase Admin SDK could not be initialized. Application may not function correctly.")

# --- Define client-side Firebase Web App Configuration ---
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
logging.info(f"Firebase Web Client Config prepared: {FIREBASE_WEB_CLIENT_CONFIG_JSON}")


# --- Helper function for robust timezone conversion ---
def normalize_to_harare_time(dt_object):
    """
    Reliably converts a datetime object (from Firestore or Python) to Harare's timezone.
    Handles Firestore Timestamps, naive datetimes (assuming UTC), and aware datetimes.
    """
    if dt_object is None:
        return None
    
    if hasattr(dt_object, 'to_datetime'):
        dt_object = dt_object.to_datetime()

    if dt_object.tzinfo is None:
        return utc_tz.localize(dt_object).astimezone(harare_tz)
    
    return dt_object.astimezone(harare_tz)


# --- Decorators for authentication and authorization checks ---
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
        if 'firebase_uid' not in session:
            flash("You need to be logged in to access this page.", "error")
            return redirect(url_for('login'))

        user_doc = get_user_doc_ref(session['firebase_uid']).get()
        if not user_doc.exists or not user_doc.to_dict().get('is_subscribed', False):
            flash("This feature requires an active subscription. Please subscribe to access.", "error")
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

# --- Firestore Path Helper Functions ---
def get_app_id_for_firestore():
    app_id = os.getenv('__app_id') or os.getenv('RENDER_SERVICE_ID') or 'default-app-id'
    return app_id

def get_public_prediction_doc_ref():
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('public_predictions').document('current_prediction')

def get_public_prediction_history_ref():
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('public_predictions_history')

def get_public_banner_doc_ref():
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('public_settings').document('banner_message')

def get_strategy_insights_doc_ref():
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('public_settings').document('strategy_insights')

def get_user_doc_ref(firebase_uid):
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('users').document(firebase_uid)

def get_user_predictions_history_ref(firebase_uid):
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('users').document(firebase_uid).collection('predictions_history')

def get_draws_collection_ref():
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('public_draw_results')

# --- Telegram Integration ---
def send_telegram_message(message, chat_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_API_URL or not chat_id:
        logging.error("Telegram not configured or chat_id missing.")
        return False, "Telegram not configured."
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
    """Stores unique draws into Firestore."""
    if db is None: return 0
    draws_collection = get_draws_collection_ref()
    inserted_count = 0
    for timestamp, mains, b1, b2, draw_type in draws_data:
        draw_date_str = timestamp.strftime('%Y-%m-%d')
        query = draws_collection.where(filter=FieldFilter('draw_date', '==', draw_date_str)).where(filter=FieldFilter('draw_type', '==', draw_type)).limit(1).get()
        if not list(query):
            try:
                draws_collection.add({
                    'main1': mains[0], 'main2': mains[1], 'main3': mains[2], 'main4': mains[3], 'main5': mains[4],
                    'bonus1': b1, 'bonus2': b2,
                    'draw_date': draw_date_str, 'draw_type': draw_type, 'timestamp': timestamp
                })
                inserted_count += 1
            except Exception as e:
                logging.error(f"Failed to add draw to Firestore: {e}")
    return inserted_count

def get_historical_draws_from_firestore(limit=None, start_date=None, end_date=None, history_window_days=None):
    """Fetches historical draws from Firestore."""
    if db is None: return []
    draws_collection = get_draws_collection_ref()
    query = draws_collection.order_by('timestamp', direction=firestore.Query.DESCENDING)
    if history_window_days:
        start_date_for_window = datetime.now(harare_tz) - timedelta(days=history_window_days)
        query = query.where(filter=FieldFilter('timestamp', '>=', start_date_for_window))
    # (Existing date filtering logic can be kept if needed)
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
    except Exception as e:
        logging.error(f"Failed to fetch historical draws: {e}")
    return draws

def get_latest_actual_draw():
    """Fetches the single latest actual draw from Firestore."""
    draws = get_historical_draws_from_firestore(limit=1)
    if draws:
        return draws[0]
    return None, None, None, None, None

# --- COMBINATORIAL ANALYSIS STRATEGY v9 ---

def super_hybrid_pool(bonus):
    """Generates a pool of candidate numbers based on the super hybrid strategy."""
    mirror = 50 - bonus
    pool = list(set([bonus, mirror, bonus + 1, bonus - 1, bonus + 10, bonus - 10]))
    return [n for n in pool if 1 <= n <= 50]

def get_hot_numbers(historical_draws, window=15):
    """Identifies numbers that have appeared frequently in a recent window."""
    if not historical_draws or len(historical_draws) < window: return []
    recent_draws = historical_draws[:window]
    recent_mains = [num for _, mains, _, _, _ in recent_draws for num in mains]
    freq = Counter(recent_mains)
    return [num for num, count in freq.items() if count > 1]

def get_cold_numbers(historical_draws, window=25):
    """Identifies numbers that have NOT appeared in a recent window."""
    if not historical_draws or len(historical_draws) < window: return []
    recent_draws = historical_draws[:window]
    recent_mains = {num for _, mains, _, _, _ in recent_draws for num in mains}
    all_possible_numbers = set(range(1, 51))
    return list(all_possible_numbers - recent_mains)

def get_recency_weighted_frequencies(historical_draws):
    """Calculates frequency of numbers, giving more weight to recent draws."""
    frequencies = defaultdict(float)
    total_draws = len(historical_draws)
    for i, draw in enumerate(historical_draws):
        weight = (total_draws - i) / total_draws
        for num in draw[1]:
            frequencies[num] += weight
    return frequencies

def get_pairing_frequencies(guaranteed_pool, historical_draws):
    """Finds numbers that frequently appear alongside numbers in the guaranteed pool."""
    pair_counts = defaultdict(int)
    for _, mains, _, _, _ in historical_draws:
        mains_set = set(mains)
        if not set(guaranteed_pool).isdisjoint(mains_set):
            for num in mains_set:
                if num not in guaranteed_pool:
                    pair_counts[num] += 1
    return pair_counts

def get_following_numbers_pool(base_mains, historical_draws):
    """
    Finds which numbers have historically appeared in the draw *immediately following*
    a draw that contained any of the numbers in `base_mains`.
    """
    following_numbers = []
    base_mains_set = set(base_mains)
    for i in range(len(historical_draws) - 1):
        current_mains = set(historical_draws[i][1])
        previous_mains = set(historical_draws[i+1][1])
        if not base_mains_set.isdisjoint(previous_mains):
            following_numbers.extend(list(current_mains))
    return [num for num, count in Counter(following_numbers).most_common(10)]

def get_trio_frequencies(historical_draws):
    """Counts the occurrences of all 3-number combinations in historical draws."""
    trio_counts = Counter()
    for _, mains, _, _, _ in historical_draws:
        for trio in itertools.combinations(sorted(mains), 3):
            trio_counts[trio] += 1
    return trio_counts

def predict_strategy(base_mains, bonus1, bonus2, historical_draws, target_size=4):
    """
    Generates a 4-number prediction using combinatorial analysis (v9)
    to maximize the chance of hitting 4 numbers.
    """
    # === Phase 1: Individual Number Scoring ===
    scores = defaultdict(float)

    # 1. Following Numbers Pool (Reduced influence, not a guarantee)
    following_pool = get_following_numbers_pool(base_mains, historical_draws)[:5]
    for num in following_pool:
        scores[num] += 250

    # 2. Pairing Frequencies (Based on the broader following_pool)
    pair_freqs = get_pairing_frequencies(following_pool, historical_draws)
    for num, freq in pair_freqs.items():
        scores[num] += freq * 75

    # 3. Hybrid, Hot, Cold Pools
    hybrid_pool = set(super_hybrid_pool(bonus1) + super_hybrid_pool(bonus2))
    hot_numbers = set(get_hot_numbers(historical_draws, window=15))
    cold_numbers = set(get_cold_numbers(historical_draws, window=25))

    for num in range(1, 51):
        boost = 0
        in_hybrid = num in hybrid_pool
        in_hot = num in hot_numbers
        in_cold = num in cold_numbers
        
        if in_hybrid: boost += 85
        if in_hot: boost += 75
        if in_cold: boost += 55
        
        # Increased overlap bonus for versatile numbers
        if sum([in_hybrid, in_hot, in_cold]) >= 2:
            boost += 150
        scores[num] += boost

    # 4. Recency-weighted frequencies
    recency_freqs = get_recency_weighted_frequencies(historical_draws)
    for num, freq in recency_freqs.items():
        scores[num] += freq * 15

    # === Phase 2: Combinatorial Analysis ===

    # 1. Get historical trio data. We only care about trios that have appeared more than once.
    trio_freqs = get_trio_frequencies(historical_draws)
    successful_trios = {trio for trio, count in trio_freqs.items() if count > 1}

    # 2. Select a pool of top candidates from Phase 1 scoring
    candidate_pool = [num for num, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:16]]

    if len(candidate_pool) < target_size:
        return sorted(candidate_pool)

    # 3. Find the best 4-number combination from the candidate pool
    best_combo = None
    max_combo_score = -1
    TRIO_BONUS = 500  # Significant bonus for forming a historically successful trio

    for combo in itertools.combinations(candidate_pool, target_size):
        combo_score = sum(scores[num] for num in combo)
        
        # Check for successful trios within this 4-number combo
        num_successful_trios = 0
        for trio in itertools.combinations(combo, 3):
            if tuple(sorted(trio)) in successful_trios:
                num_successful_trios += 1
        
        combo_score += num_successful_trios * TRIO_BONUS

        if combo_score > max_combo_score:
            max_combo_score = combo_score
            best_combo = combo

    return sorted(list(best_combo)) if best_combo else sorted(candidate_pool[:target_size])


def generate_live_prediction(historical_draws):
    """
    Generates the current live prediction using the most recent draw and historical data.
    """
    if not historical_draws or len(historical_draws) < 3:
        logging.warning("Not enough historical draws to generate a live prediction.")
        return None

    latest_draw = historical_draws[0]
    base_mains = latest_draw[1]
    bonus1, bonus2 = latest_draw[2], latest_draw[3]

    prediction = predict_strategy(base_mains, bonus1, bonus2, historical_draws[1:])
    
    return {
        'strategy_used': 'combinatorial_analysis_v9',
        'prediction': prediction
    }


def backtest_strategy(draws_data):
    """
    Performs a true backtest on historical data using the v9 model.
    """
    logging.debug(f"backtest_strategy received {len(draws_data)} draws for processing.")
    results = []
    # Need at least 50 draws for a valid backtest window.
    if len(draws_data) < 50:
        logging.info(f"Not enough draws for a valid backtest (need at least 50). Got {len(draws_data)}. Skipping.")
        return [], [], []

    # This loop iterates through all possible draws for which a valid prediction can be made.
    for i in range(len(draws_data) - 27):
        target_actual_draw = draws_data[i]
        base_draw_for_prediction = draws_data[i + 1]
        historical_data_for_prediction = draws_data[i + 2:]

        base_draw_mains = base_draw_for_prediction[1]
        base_draw_b1 = base_draw_for_prediction[2]
        base_draw_b2 = base_draw_for_prediction[3]
        
        target_timestamp, target_mains, _, _, target_draw_type = target_actual_draw
        
        # Generate the prediction using the v9 strategy.
        predicted_mains = predict_strategy(base_draw_mains, base_draw_b1, base_draw_b2, historical_data_for_prediction)
        
        hits = len(set(predicted_mains).intersection(set(target_mains)))

        results.append({
            'target_draw_time': target_timestamp,
            'draw_date': target_timestamp.strftime('%Y-%m-%d'),
            'draw_type': target_draw_type,
            'strategy_used': 'combinatorial_analysis_v9',
            'bonus': [base_draw_b1, base_draw_b2],
            'prediction': predicted_mains,
            'actual_mains': target_mains,
            'actual_bonuses': [target_actual_draw[2], target_actual_draw[3]],
            'hits': hits
        })

    main_ranking = Counter(num for _, mains, _, _, _ in draws_data for num in mains).most_common()
    bonus_ranking = Counter(b for _, _, b1, b2, _ in draws_data for b in (b1, b2)).most_common()
    return results, main_ranking, bonus_ranking


# --- Data Class for Prediction Results ---
class PredictionResult:
    def __init__(self, target_draw_time, strategy_used, bonus, prediction, actual_mains=None, actual_bonuses=None, hits=None):
        self.target_draw_time = target_draw_time
        self.strategy_used = strategy_used
        self.bonus = bonus
        self.prediction = prediction
        self.actual_mains = actual_mains
        self.actual_bonuses = actual_bonuses
        self.hits = hits

    def to_dict(self):
        return vars(self)

def get_next_target_draw_time(current_time_harare):
    """Calculates the next official Gosloto 5/50 draw time."""
    now = current_time_harare.astimezone(harare_tz).replace(second=0, microsecond=0)
    today_morning = now.replace(hour=8, minute=30)
    today_evening = now.replace(hour=20, minute=30)
    if now < today_morning: return today_morning
    if now < today_evening: return today_evening
    return (now + timedelta(days=1)).replace(hour=8, minute=30)

# --- Flask Routes ---
@app.context_processor
def inject_global_data():
    is_subscribed = False
    if 'firebase_uid' in session and db:
        try:
            user_doc = get_user_doc_ref(session['firebase_uid']).get()
            if user_doc.exists: is_subscribed = user_doc.to_dict().get('is_subscribed', False)
        except Exception as e:
            logging.error(f"Error fetching user subscription status: {e}")
    banner_message = "Welcome!"
    if db:
        try:
            banner_doc = get_public_banner_doc_ref().get()
            if banner_doc.exists: banner_message = banner_doc.to_dict().get('message', banner_message)
        except Exception as e:
            logging.error(f"Error fetching banner message: {e}")
    return dict(logged_in='firebase_uid' in session, is_subscribed=is_subscribed, is_admin=session.get('is_admin', False),
                banner_message=banner_message, now=datetime.now(harare_tz), firebase_web_client_config_json=FIREBASE_WEB_CLIENT_CONFIG_JSON)

@app.route('/')
def index():
    if db is None:
        flash("Database not initialized.", "error")
        return render_template('index.html', error="Database connection error.")
    
    latest_draw_time, latest_mains, b1, b2, _ = get_latest_actual_draw()
    latest_bonuses = [b1, b2] if b1 else []
    
    live_prediction_history = []
    current_prediction_vs_actual = None

    try:
        public_prediction_doc = get_public_prediction_doc_ref().get()
        if public_prediction_doc.exists:
            pred_data = public_prediction_doc.to_dict()
            unresolved = PredictionResult(normalize_to_harare_time(pred_data.get('target_draw_time')), pred_data.get('strategy_used'), pred_data.get('bonus'), pred_data.get('prediction'))
            live_prediction_history.append(unresolved)
    except Exception as e:
        logging.error(f"Could not fetch public prediction: {e}")

    if session.get('is_subscribed'):
        try:
            history_docs = get_user_predictions_history_ref(session['firebase_uid']).order_by('target_draw_time', direction=firestore.Query.DESCENDING).limit(10).stream()
            for doc in history_docs:
                data = doc.to_dict()
                pred_time = normalize_to_harare_time(data.get('target_draw_time'))
                if not live_prediction_history or pred_time != live_prediction_history[0].target_draw_time:
                    live_prediction_history.append(PredictionResult(pred_time, data.get('strategy_used'), data.get('bonus'), data.get('prediction'), data.get('actual_mains'), data.get('actual_bonuses'), data.get('hits')))
        except Exception as e:
            logging.error(f"Error fetching prediction history: {e}")

    if live_prediction_history:
        current_prediction_vs_actual = live_prediction_history[0]

    combos_3, combos_2 = [], []
    if current_prediction_vs_actual and current_prediction_vs_actual.prediction:
        combos_3 = list(itertools.combinations(current_prediction_vs_actual.prediction, 3))
        combos_2 = list(itertools.combinations(current_prediction_vs_actual.prediction, 2))

    return render_template('index.html', latest_draw_time=latest_draw_time, latest_actual_mains=latest_mains, latest_actual_bonuses=latest_bonuses,
                           current_prediction_vs_actual=current_prediction_vs_actual, generated_main_combos_3=combos_3,
                           generated_main_combos_2=combos_2, live_prediction_history=live_prediction_history)

@app.route('/backtest_results', methods=['GET', 'POST'])
@login_required
@admin_required
@subscription_required
def backtest_results_page():
    start_date = request.form.get('start_date') if request.method == 'POST' else None
    end_date = request.form.get('end_date') if request.method == 'POST' else None
    historical_draws = get_historical_draws_from_firestore(start_date=start_date, end_date=end_date)
    backtest_results, main_ranking, bonus_ranking = [], [], []
    if historical_draws:
        backtest_results, main_ranking, bonus_ranking = backtest_strategy(historical_draws)
    
    total_hits = sum(r['hits'] for r in backtest_results)
    total_predicted = sum(len(r.get('prediction', [])) for r in backtest_results)
    avg_hit_rate = (total_hits / total_predicted) * 100 if total_predicted > 0 else 0
    
    return render_template('backtest_results.html', backtest_results=backtest_results, total_backtests=len(backtest_results),
                           average_hit_rate=avg_hit_rate, total_4_hits=sum(1 for r in backtest_results if r['hits'] == 4),
                           total_3_hits=sum(1 for r in backtest_results if r['hits'] >= 3), total_2_hits=sum(1 for r in backtest_results if r['hits'] >= 2),
                           total_1_hits=sum(1 for r in backtest_results if r['hits'] >= 1), total_0_hits=sum(1 for r in backtest_results if r['hits'] == 0),
                           main_ranking=main_ranking, bonus_ranking=bonus_ranking, selected_start_date=start_date, selected_end_date=end_date)

# --- Auth Routes (Login, Register, Logout) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id_token = request.json.get('idToken')
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            session['firebase_uid'] = uid
            session['is_admin'] = decoded_token.get('admin', False)
            user_doc = get_user_doc_ref(uid).get()
            if user_doc.exists:
                session['is_subscribed'] = user_doc.to_dict().get('is_subscribed', False)
            else: # Create user doc on first login
                get_user_doc_ref(uid).set({'email': decoded_token.get('email'), 'is_subscribed': False}, merge=True)
                session['is_subscribed'] = False
            return jsonify(success=True, redirect=url_for('index'))
        except Exception as e:
            return jsonify(success=False, message=str(e)), 401
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
            # Auto-admin for the very first user
            users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
            users_count = users_ref.limit(2).get()
            if len(users_count) == 1:
                auth.set_custom_user_claims(uid, {'admin': True})
                session['is_admin'] = True
            session['firebase_uid'] = uid
            return jsonify(success=True, redirect=url_for('index'))
        except Exception as e:
            return jsonify(success=False, message=str(e)), 400
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# --- User and Admin Routes ---
@app.route('/subscribe_info')
@login_required
def subscribe_info():
    return render_template('subscribe_info.html')

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

@app.route('/admin_panel')
@login_required
@admin_required
def admin_panel():
    return render_template('admin_panel.html')

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
            except Exception as e:
                flash(f"Error updating banner: {e}", "error")
        return redirect(url_for('admin_banner_settings'))

    current_message = "Welcome!"
    try:
        banner_doc = banner_doc_ref.get()
        if banner_doc.exists:
            current_message = banner_doc.to_dict().get('message', current_message)
    except Exception as e:
        flash(f"Error fetching current banner message: {e}", "error")

    return render_template('admin_banner_settings.html', current_message=current_message)

@app.route('/set_admin_claim', methods=['GET', 'POST'])
@login_required
@admin_required
def set_admin_claim():
    if request.method == 'POST':
        email = request.form.get('email')
        action = request.form.get('action') # 'grant' or 'revoke'
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
            except Exception as e:
                logging.warning(f"Could not fetch claims for user {uid}: {e}")
            users_list.append({**user_data, 'uid': uid, 'is_admin': is_admin})
    except Exception as e:
        flash(f"Error fetching users: {e}", "error")
    return render_template('admin_users.html', users=users_list)

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
            flash(f"Subscription for {uid} set to {new_status}.", "success")
    except Exception as e:
        flash(f"Error toggling subscription: {e}", "error")
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

@app.route('/admin_send_results', methods=['POST'])
@login_required
@admin_required
def admin_send_results():
    """Manually triggers the job to send results and hits to users."""
    try:
        send_results_and_hits_job()
        flash("Results and hits notifications sent successfully!", "success")
    except Exception as e:
        flash(f"Error sending results notifications: {e}", "error")
        logging.error(f"Error manually sending results notifications: {e}")
    return redirect(url_for('admin_panel'))


# --- Scheduler and Jobs ---
scheduler = BackgroundScheduler(daemon=True, timezone=harare_tz)
HISTORY_WINDOW_DAYS = 60

def scrape_and_process_draws_job():
    """Main job to scrape, store, predict, and update."""
    logging.info("--- JOB START: Scrape and Process Draws ---")
    
    # 1. Scrape and Store Draws
    draws_data, error = fetch_draws_from_website()
    if error:
        logging.error(f"Scraping job failed: {error}")
        return
    store_draws_to_firestore(draws_data)

    # 2. Generate Live Prediction
    historical_draws = get_historical_draws_from_firestore(history_window_days=HISTORY_WINDOW_DAYS)
    live_prediction_result = generate_live_prediction(historical_draws)

    if live_prediction_result:
        predicted_mains = live_prediction_result['prediction']
        strategy_used = live_prediction_result['strategy_used']
        latest_actual_bonus1 = historical_draws[0][2]
        latest_actual_bonus2 = historical_draws[0][3]
        next_target_draw_time = get_next_target_draw_time(datetime.now(harare_tz))

        try:
            # Save the new public prediction
            get_public_prediction_doc_ref().set({
                'target_draw_time': next_target_draw_time,
                'strategy_used': strategy_used,
                'bonus': [latest_actual_bonus1, latest_actual_bonus2],
                'prediction': predicted_mains,
                'actual_mains': [], 'actual_bonuses': [], 'hits': 0,
                'timestamp_generated': firestore.SERVER_TIMESTAMP
            }, merge=True)
            logging.info(f"Saved new public prediction: {predicted_mains}")

            # Save to public history for repetition checks
            get_public_prediction_history_ref().add({
                'prediction': predicted_mains,
                'target_draw_time': next_target_draw_time,
                'timestamp_generated': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            logging.error(f"Failed to save public prediction: {e}")
    else:
        logging.error("Failed to generate live prediction. Job cannot continue.")
        return # Stop if no prediction was made

    # 3. Update user predictions and check past hits
    update_all_user_predictions_job()
    check_and_update_prediction_hits_job()
    logging.info("--- JOB END: Scrape and Process Draws ---")

def update_all_user_predictions_job():
    """Updates predictions for all subscribed users based on the public prediction."""
    public_pred_doc = get_public_prediction_doc_ref().get()
    if not public_pred_doc.exists:
        logging.warning("No public prediction found to update users with.")
        return
    pred_data = public_pred_doc.to_dict()
    target_time = normalize_to_harare_time(pred_data.get('target_draw_time'))
    pred_doc_id = target_time.strftime('%Y-%m-%d_%H%M')

    users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
    subscribed_users = users_ref.where(filter=FieldFilter('is_subscribed', '==', True)).stream()
    for user in subscribed_users:
        history_ref = get_user_predictions_history_ref(user.id)
        if not history_ref.document(pred_doc_id).get().exists:
            history_ref.document(pred_doc_id).set(pred_data, merge=True)
            logging.info(f"Saved prediction for user {user.id}")

def check_and_update_prediction_hits_job():
    """Checks and updates hits for past predictions."""
    latest_draw_time, latest_mains, b1, b2, _ = get_latest_actual_draw()
    if not latest_draw_time or not latest_mains: return

    target_time = latest_draw_time.replace(second=0, microsecond=0)
    target_time = target_time.replace(hour=8, minute=30) if target_time.hour < 12 else target_time.replace(hour=20, minute=30)
    pred_doc_id = target_time.strftime('%Y-%m-%d_%H%M')

    users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
    for user_doc in users_ref.stream():
        pred_ref = get_user_predictions_history_ref(user_doc.id).document(pred_doc_id)
        prediction = pred_ref.get()
        if prediction.exists and not prediction.to_dict().get('actual_mains'):
            predicted_mains = prediction.to_dict().get('prediction', [])
            hits = len(set(predicted_mains).intersection(set(latest_mains)))
            pred_ref.update({'actual_mains': latest_mains, 'actual_bonuses': [b1, b2], 'hits': hits})
            logging.info(f"Updated hits for user {user_doc.id}: {hits} hits.")

def precompute_successful_bonuses_job():
    """Pre-computes and stores bonuses that have historically led to 3+ hits."""
    logging.info("--- JOB START: Pre-compute Successful Bonuses ---")
    historical_draws = get_historical_draws_from_firestore(history_window_days=180)
    if len(historical_draws) < 20: return
    
    backtest_results, _, _ = backtest_strategy(historical_draws)
    successful_bonuses = set()
    for result in backtest_results:
        if result.get('hits', 0) >= 3:
            try:
                # This part might need adjustment depending on the new strategy_used string format
                bonus_str = result.get('strategy_used', '').split('_')[3] # Example, might need change
                bonus_num = int(bonus_str)
                successful_bonuses.add(bonus_num)
            except (ValueError, IndexError):
                continue
    
    if successful_bonuses:
        get_strategy_insights_doc_ref().set({
            'bonuses_with_3_hits': sorted(list(successful_bonuses)),
            'last_updated': firestore.SERVER_TIMESTAMP
        })
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
    """Sends upcoming prediction alerts to subscribed Telegram users."""
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
    """Sends draw results and user-specific hits to Telegram."""
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

# --- Schedule All Jobs ---
scheduler.add_job(scrape_and_process_draws_job, 'cron', hour='8,20', minute='40,45,50,55', id='scrape_process_job', replace_existing=True)
scheduler.add_job(precompute_successful_bonuses_job, 'cron', hour='3', minute='0', id='precompute_bonuses_job', replace_existing=True)
scheduler.add_job(send_prediction_alerts_job, 'cron', hour='7,19', minute='30', id='send_prediction_alerts_job', replace_existing=True)
scheduler.add_job(send_results_and_hits_job, 'cron', hour='8,20', minute='35,40', id='send_results_hits_job', replace_existing=True)


if __name__ == '__main__':
    # Start the scheduler. In a production environment (like Render), the web server 
    # (e.g., Gunicorn) runs the app, and the scheduler starts as part of the app process.
    scheduler.start()
    # Shut down the scheduler when the app exits
    atexit.register(lambda: scheduler.shutdown())
    # The line below is for local development. For production, use a WSGI server.
    # app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
