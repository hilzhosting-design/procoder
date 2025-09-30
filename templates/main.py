import os
import json
import sys
from datetime import datetime, timedelta
import pytz
import requests, re, random
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
                # Corrected: Use initialize_app instead of initializeApp
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
# This configuration is crucial for the client-side JavaScript to interact with Firebase.
# It should be populated with your actual Firebase project's web app configuration.
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
    
    # If it's a Firestore Timestamp, convert it to a Python datetime
    if hasattr(dt_object, 'to_datetime'):
        dt_object = dt_object.to_datetime()

    # If the datetime object is naive, assume it's UTC, make it aware, then convert
    if dt_object.tzinfo is None:
        return utc_tz.localize(dt_object).astimezone(harare_tz)
    
    # If it's already aware, just convert it to the target timezone
    return dt_object.astimezone(harare_tz)


# --- Decorators for authentication and authorization checks ---
def login_required(f):
    """
    Decorator to ensure a user is logged in before accessing a route.
    Redirects to login page if not logged in.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'firebase_uid' not in session:
            flash("You need to be logged in to access this page.", "error")
            logging.debug(f"Login required for {request.path}. Redirecting to login.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def subscription_required(f):
    """
    Decorator to ensure a user is subscribed before accessing a route.
    Redirects to subscribe info page if not subscribed.
    Assumes login_required has already been applied or user is known to be logged in.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'firebase_uid' not in session: # Should ideally be caught by login_required first
            flash("You need to be logged in to access this page.", "error")
            logging.debug(f"Subscription required but user not logged in for {request.path}. Redirecting to login.")
            return redirect(url_for('login'))

        user_doc = get_user_doc_ref(session['firebase_uid']).get()
        if not user_doc.exists or not user_doc.to_dict().get('is_subscribed', False):
            flash("This feature requires an active subscription. Please subscribe to access.", "error")
            logging.debug(f"Subscription required for {request.path}. User {session['firebase_uid']} not subscribed. Redirecting to subscribe info.")
            return redirect(url_for('subscribe_info'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """
    Decorator to ensure the logged-in user has admin privileges (based on custom claim).
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logging.debug(f"Checking admin status for user {session.get('firebase_uid')}: is_admin={session.get('is_admin', False)} for route {request.path}.")

        if not session.get('is_admin', False):
            flash("Access denied. Admin privileges required.", "error")
            logging.warning(f"Non-admin user {session.get('firebase_uid')} attempted to access admin route: {request.path}.")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
# --- End of Decorators ---


def get_app_id_for_firestore():
    """
    Determines the app ID to use for Firestore paths.
    Prioritizes __app_id (Canvas), then RENDER_SERVICE_ID, then a default.
    """
    app_id = os.getenv('__app_id') # Canvas specific
    if app_id:
        logging.debug(f"Using __app_id from environment: {app_id}")
        return app_id

    app_id = os.getenv('RENDER_SERVICE_ID') # Render service ID
    if app_id:
        logging.debug(f"Using RENDER_SERVICE_ID from environment: {app_id}")
        return app_id

    logging.warning("Neither __app_id nor RENDER_SERVICE_ID found. Falling back to 'default-app-id'.")
    return 'default-app-id'


def get_current_user_id():
    """Retrieves the current user's Firebase UID from the session."""
    if has_request_context():
        return session.get('firebase_uid')
    else:
        logging.critical("get_current_user_id called outside of request context. This should not happen in automation mode.")
        return 'system_automation_user'

def get_public_prediction_doc_ref():
    """Returns the Firestore DocumentReference for the public prediction."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get public prediction reference.")
        raise RuntimeError("Firestore DB client not initialized.")
    app_id = get_app_id_for_firestore()
    # Corrected path to be consistent with the new structure: artifacts/{appId}/public_predictions/current_prediction
    return db.collection('artifacts').document(app_id).collection('public_predictions').document('current_prediction')

def get_public_prediction_history_ref():
    """Returns the Firestore CollectionReference for the public prediction history."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get public prediction history reference.")
        raise RuntimeError("Firestore DB client not initialized.")
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('public_predictions_history')


def get_public_banner_doc_ref():
    """Returns the Firestore DocumentReference for the public banner settings."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get public banner reference.")
        raise RuntimeError("Firestore DB client not initialized.")
    app_id = get_app_id_for_firestore()
    # Corrected path to be consistent with the new structure: artifacts/{appId}/public_settings/banner_message
    return db.collection('artifacts').document(app_id).collection('public_settings').document('banner_message')

def get_strategy_insights_doc_ref():
    """Returns the Firestore DocumentReference for storing pre-computed strategy insights."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get strategy insights reference.")
        raise RuntimeError("Firestore DB client not initialized.")
    app_id = get_app_id_for_firestore()
    # This document will store our pre-computed successful bonuses.
    return db.collection('artifacts').document(app_id).collection('public_settings').document('strategy_insights')

def get_user_doc_ref(firebase_uid):
    """Returns the Firestore DocumentReference for a specific user."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get user document reference.")
        raise RuntimeError("Firestore DB client not initialized.")
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('users').document(firebase_uid)

def get_user_predictions_history_ref(firebase_uid):
    """Returns the Firestore CollectionReference for a user's prediction history."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get user predictions history reference.")
        raise RuntimeError("Firestore DB client not initialized.")
    app_id = get_app_id_for_firestore()
    return db.collection('artifacts').document(app_id).collection('users').document(firebase_uid).collection('predictions_history')

def get_draws_collection_ref():
    """Returns the Firestore CollectionReference for historical draws."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot get draws collection reference.")
        return None # Return None instead of raising an error to allow graceful handling
    app_id = get_app_id_for_firestore()
    # Corrected path to be consistent with the new structure: artifacts/{appId}/public_draw_results
    return db.collection('artifacts').document(app_id).collection('public_draw_results')


def send_telegram_message(message, chat_id):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_API_URL:
        logging.error("Telegram BOT_TOKEN not set or API URL not constructed.")
        return False, "Telegram bot not configured. Please set BOT_TOKEN."
    if not chat_id:
        logging.error("Telegram CHAT_ID is missing for the user." )
        return False, "Your Telegram Chat ID is not set. Please go to Telegram Settings to set it up."

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        response = requests.post(TELEGRAM_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        logging.debug(f"Telegram message sent successfully to chat ID {chat_id}: {message[:50]}...")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Telegram message to chat ID {chat_id}: {e}")
        return False, f"Failed to send Telegram message: {e}"
    return True, "Message sent successfully!"

# Using the URL from your provided main (29).py
GOSLOTO_5_50_URL = "https://www.comparethelotto.com/za/gosloto-5-50-results" # Updated URL to comparethelotto.com

def fetch_draws_from_website():
    """
    Fetches Gosloto 5/50 draw results from comparethelotto.com.
    Returns a list of tuples: (datetime_object_harare, main_numbers_list, bonus1_int, bonus2_int, draw_type_str)
    Includes retry mechanism.
    """
    url = GOSLOTO_5_50_URL
    logging.debug(f"Attempting to fetch draws from: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            logging.debug(f"Successfully fetched URL. Status Code: {response.status_code}")

            soup = BeautifulSoup(response.text, "html.parser")

            # Find the main container for historical results
            historical_results_body = soup.select_one("div#historicalResults > div.card-body")

            if not historical_results_body:
                logging.warning("Could not find the main historical results container (div#historicalResults > div.card-body). HTML structure might have changed.")
                if attempt < retries - 1:
                    logging.info(f"Retrying in 5 seconds (attempt {attempt + 1}/{retries})...")
                    time.sleep(5)
                    continue
                return [], "No historical results container found on the page after multiple attempts."

            raw_draws_with_timestamps = []

            # --- REVISED, MORE ROBUST PARSING LOGIC ---

            # 1. Find all <p> tags that mark the beginning of each draw.
            draw_start_points = historical_results_body.find_all('p', class_='text-muted')

            if not draw_start_points:
                logging.warning("No draw start points ('p.text-muted') found inside the container. The website structure may have changed.")
                return [], "No individual draws found on the page."

            logging.debug(f"Found {len(draw_start_points)} potential draw start points to process.")

            # 2. Loop through each starting point and parse its specific draw data.
            for date_p_tag in draw_start_points:
                # First, parse the date and time from the <p> tag itself.
                full_date_time_text = date_p_tag.get_text(separator=' ').strip()
                clean_date_time_text = re.sub(r'<[^>]+>', '', full_date_time_text)
                clean_date_time_text = re.sub(r'\s*&nbsp;\s*', ' ', clean_date_time_text).strip()
                clean_date_time_text = re.sub(r'^\w{3}\s+', '', clean_date_time_text)
                clean_date_time_text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', clean_date_time_text)
                clean_date_time_text = re.sub(r'\s*\([^)]*\)', '', clean_date_time_text).strip()
                clean_date_time_text = re.sub(r'\s+NEW$', '', clean_date_time_text).strip()

                timestamp = None
                try:
                    naive_timestamp = datetime.strptime(clean_date_time_text, '%d %B %Y %H:%M')
                    timestamp = harare_tz.localize(naive_timestamp)
                except ValueError as ve:
                    logging.error(f"Could not parse timestamp from text: '{clean_date_time_text}'. Error: {ve}. Skipping this block.")
                    continue

                main_numbers = []
                bonus_numbers = []

                # 3. Iterate through the elements immediately following the date tag to collect numbers.
                for sibling in date_p_tag.next_siblings:
                    if sibling.name is None:  # Skip non-tag elements like whitespace.
                        continue

                    # Stop collecting numbers if we hit the next date tag or a horizontal rule.
                    if sibling.name == 'p' or sibling.name == 'hr':
                        break

                    # If the sibling is a <span>, check if it's a number we need.
                    if sibling.name == 'span':
                        classes = sibling.get('class', [])
                        if 'results-number' in classes:
                            try:
                                num = int(sibling.text.strip())
                                main_numbers.append(num)
                            except (ValueError, TypeError):
                                pass  # Ignore if text is not a valid number.
                        elif 'results-number-bonus' in classes:
                            try:
                                num = int(sibling.text.strip())
                                bonus_numbers.append(num)
                            except (ValueError, TypeError):
                                pass

                # 4. After collecting all numbers for this draw, validate and store it.
                if len(main_numbers) == 5 and len(bonus_numbers) == 2:
                    draw_type = "Morning" if timestamp.hour < 12 else "Evening"
                    raw_draws_with_timestamps.append((timestamp, sorted(main_numbers), bonus_numbers[0], bonus_numbers[1], draw_type))
                    logging.info(f"Successfully parsed draw: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                else:
                    logging.warning(f"Found incomplete draw for timestamp {timestamp.strftime('%Y-%m-%d %H:%M')}. Mains: {len(main_numbers)}, Bonuses: {len(bonus_numbers)}. Skipping.")

            # --- END OF REVISED LOGIC ---

            if raw_draws_with_timestamps:
                raw_draws_with_timestamps.sort(key=lambda x: x[0], reverse=True)

            unique_draws = []
            seen_draw_identifiers = set()
            for timestamp, main_nums, b1, b2, draw_type in raw_draws_with_timestamps:
                draw_date_str = timestamp.strftime('%Y-%m-%d')

                draw_identifier = (draw_date_str, draw_type)
                if draw_identifier not in seen_draw_identifiers:
                    unique_draws.append((timestamp, main_nums, b1, b2, draw_type))
                    seen_draw_identifiers.add(draw_identifier)
                else:
                    logging.debug(f"Skipping duplicate draw (date/type): {draw_date_str} {draw_type} during uniqueness check.")

            logging.info(f"fetch_draws_from_website returning {len(unique_draws)} unique, sorted 5/50 draws.")
            return unique_draws, None

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed to fetch draws from {url}: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in 5 seconds (attempt {attempt + 1}/{retries})...")
                time.sleep(5)
            else:
                return [], f"Error fetching draw data after {retries} attempts: {e}. Please check your internet connection or the website."
    return [], "Unexpected error: Should not reach here." # Fallback in case loop finishes without return

def store_draws_to_firestore(draws_data):
    """Stores unique draws into Firestore's 'draws' collection."""
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot store draws.")
        return 0

    draws_collection = get_draws_collection_ref()
    inserted_count = 0
    logging.debug(f"Attempting to store {len(draws_data)} draws to Firestore. Relevant app_id: {get_app_id_for_firestore()}")

    for timestamp, mains, b1, b2, draw_type in draws_data:
        draw_date_str = timestamp.strftime('%Y-%m-%d')

        # Check if the draw already exists
        query = draws_collection.where(filter=FieldFilter('draw_date', '==', draw_date_str)).where(filter=FieldFilter('draw_type', '==', draw_type)).limit(1).get()

        if not list(query): # If the draw does NOT exist, try to add it
            try:
                draws_collection.add({
                    'main1': mains[0], 'main2': mains[1], 'main3': mains[2], 'main4': mains[3], 'main5': mains[4],
                    'bonus1': b1, 'bonus2': b2,
                    'draw_date': draw_date_str,
                    'draw_type': draw_type,
                    'timestamp': timestamp
                })
                inserted_count += 1
                logging.info(f"Stored new draw to Firestore: {draw_date_str} {draw_type} - Mains: {mains}, Bonuses: {b1}, {b2}")
            except Exception as e: # Catch any errors during the add operation
                logging.error(f"Failed to add draw {draw_date_str} {draw_type} to Firestore: {e}")
        else: # If the draw already exists (list(query) is not empty)
            logging.debug(f"Draw {draw_date_str} {draw_type} already exists in Firestore. Skipping.")
    logging.info(f"Finished storing draws. Inserted {inserted_count} new draws.")
    return inserted_count

def get_historical_draws_from_firestore(limit=None, start_date=None, end_date=None, history_window_days=None):
    """
    Fetches historical draws from Firestore, most recent first, with optional date filtering
    or a sliding window based on history_window_days.
    """
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot fetch historical draws.")
        return []

    draws_collection = get_draws_collection_ref()
    query = draws_collection.order_by('timestamp', direction=firestore.Query.DESCENDING)

    if history_window_days is not None:
        # Calculate the start date for the sliding window
        end_date_for_window = datetime.now(harare_tz)
        start_date_for_window = end_date_for_window - timedelta(days=history_window_days)
        query = query.where(filter=FieldFilter('timestamp', '>=', start_date_for_window))
        logging.debug(f"Filtering draws for history window: from {start_date_for_window.strftime('%Y-%m-%d')} to {end_date_for_window.strftime('%Y-%m-%d')}")
    else:
        if start_date:
            # Ensure start_date is a datetime object in the correct timezone
            if isinstance(start_date, str):
                start_date_dt = harare_tz.localize(datetime.strptime(start_date, '%Y-%m-%d'))
            else:
                start_date_dt = start_date.astimezone(harare_tz)
            query = query.where(filter=FieldFilter('timestamp', '>=', start_date_dt))
            logging.debug(f"Filtering draws from start_date: {start_date_dt}")

        if end_date:
            # Ensure end_date is a datetime object in the correct timezone
            if isinstance(end_date, str):
                # Add one day to end_date to include the entire end day
                end_date_dt = harare_tz.localize(datetime.strptime(end_date, '%Y-%m-%d')) + timedelta(days=1)
            else:
                end_date_dt = end_date.astimezone(harare_tz) + timedelta(days=1)
            query = query.where(filter=FieldFilter('timestamp', '<', end_date_dt))
            logging.debug(f"Filtering draws up to end_date (exclusive): {end_date_dt}")

        if limit:
            query = query.limit(limit)

    draws = []
    try:
        docs = query.stream()
        for doc in docs:
            data = doc.to_dict()

            timestamp_value = data.get('timestamp')
            timestamp_dt = normalize_to_harare_time(timestamp_value) # Use helper function
            if not timestamp_dt:
                logging.error(f"Unexpected type for timestamp in Firestore: {type(timestamp_value)}. Skipping draw.")
                continue

            mains = [data.get(f'main{i}') for i in range(1, 6)]
            bonuses = [data.get('bonus1'), data.get('bonus2')]

            if any(m is None for m in mains) or any(b is None for b in bonuses):
                logging.warning(f"Skipping draw {doc.id} due to missing main or bonus numbers: {data}")
                continue

            draws.append((
                timestamp_dt,
                mains,
                bonuses[0],
                bonuses[1],
                data.get('draw_type', 'Unknown')
            ))
        logging.info(f"Fetched {len(draws)} historical draws from Firestore. Data received: {draws[:2]}...")
    except Exception as e:
        logging.error(f"Failed to fetch historical draws from Firestore: {e}")
        return []
    return draws

def get_latest_actual_draw():
    """
    Fetches the single latest actual draw from Firestore.
    Returns (timestamp_harare, mains_list, bonus1, bonus2, draw_type) or (None, None, None, None, None)
    """
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot fetch latest actual draw.")
        return None, None, None, None, None

    draws_collection = get_draws_collection_ref()
    if draws_collection is None: # Handle case where get_draws_collection_ref returns None
        return None, None, None, None, None

    try:
        query = draws_collection.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        docs = query.stream()

        for doc in docs:
            data = doc.to_dict()

            timestamp_value = data.get('timestamp')
            timestamp_dt = normalize_to_harare_time(timestamp_value) # Use helper function

            if timestamp_dt:
                mains = [data.get('main1'), data.get('main2'), data.get('main3'), data.get('main4'), data.get('main5')]
                bonuses = [data.get('bonus1'), data.get('bonus2')]
                draw_type = data.get('draw_type', 'Unknown')
                logging.debug(f"Latest actual draw fetched: {timestamp_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
                return timestamp_dt, mains, bonuses[0], bonuses[1], draw_type

        logging.info("No latest actual draw found in Firestore.")
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"Failed to fetch latest actual draw from Firestore: {e}")
        return None, None, None, None, None


def build_frequency_map(draws_data):
    """
    Builds a frequency map of main numbers based on bonus balls, applying a linear weight.
    More recent draws have higher weight.
    `draws_data` is expected to be a list of (timestamp, mains, bonus1, bonus2, draw_type) tuples,
    sorted from most recent to oldest.
    """
    freq_map = defaultdict(lambda: defaultdict(int)) # bonus -> main_number -> count
    logging.debug(f"build_frequency_map received {len(draws_data)} draws for weighted analysis.")

    # Assign weights: most recent draw gets highest weight, decreasing linearly
    # If there are N draws, the most recent gets weight N, the next N-1, and so on.
    for i, draw_item in enumerate(draws_data):
        if len(draw_item) < 5:
            logging.error(f"Malformed draw item {i} in build_frequency_map. Skipping. Item: {draw_item}")
            continue
        try:
            timestamp, mains, b1, b2, draw_type = draw_item
            weight = len(draws_data) - i # Linear weight: most recent (i=0) gets max weight, oldest (i=N-1) gets weight 1

            for main_num in mains:
                freq_map[b1][main_num] += weight
                freq_map[b2][main_num] += weight
        except ValueError as e:
            logging.error(f"Failed to unpack draw item {i} in build_frequency_map. Item: {draw_item}, Error: {e}")
            continue

    # Convert weighted counts to a list of top 10 numbers for each bonus
    result_map = {}
    for bonus, main_counts in freq_map.items():
        # Sort main numbers by their weighted count in descending order
        sorted_mains = sorted(main_counts.items(), key=lambda item: item[1], reverse=True)
        result_map[bonus] = [num for num, count in sorted_mains[:10]] # Take top 10

    logging.debug(f"Built weighted frequency map with {len(result_map)} bonus keys.")
    return result_map

def super_hybrid(bonus):
    """
    Generates a set of numbers based on the super hybrid strategy.
    """
    hybrid_set = {n for n in {bonus, 50 - bonus, bonus + 1, bonus - 1, bonus + 10, bonus - 10} if 1 <= n <= 50}
    logging.debug(f"Super hybrid for bonus {bonus}: {sorted(list(hybrid_set))}")
    return hybrid_set

def predict_strategy(bonus, freq_map):
    """
    Generates a prediction of 4 main numbers based on Hilliman-like logic combined with Super Hybrid.
    This version introduces randomness to reduce repeating predictions.
    """
    predicted_numbers_set = set()

    # 1. Start with top 3 from frequency map
    top_from_freq = freq_map.get(bonus, [])
    # Take up to 3 from the top_from_freq, prioritizing them
    for num in top_from_freq:
        if len(predicted_numbers_set) < 4:
            predicted_numbers_set.add(num)
        else:
            break

    # 2. Add numbers from the super hybrid candidates, randomly
    hybrid_candidates = super_hybrid(bonus)
    # Create a pool of hybrid candidates not already picked
    candidate_pool = list(hybrid_candidates - predicted_numbers_set)
    random.shuffle(candidate_pool) # Shuffle to introduce randomness

    # Add numbers from the shuffled hybrid pool until 4 numbers are picked
    for num in candidate_pool:
        if len(predicted_numbers_set) < 4:
            predicted_numbers_set.add(num)
        else:
            break

    # 3. If still less than 4, fill remaining slots with random numbers from 1-50 not already picked
    remaining_slots = 4 - len(predicted_numbers_set)
    if remaining_slots > 0:
        all_numbers = list(range(1, 51))
        random.shuffle(all_numbers) # Shuffle all numbers
        for num in all_numbers:
            if len(predicted_numbers_set) < 4 and num not in predicted_numbers_set:
                predicted_numbers_set.add(num)
            if len(predicted_numbers_set) == 4: # Stop once 4 numbers are picked
                break

    return sorted(list(predicted_numbers_set))

def backtest_strategy(draws_data):
    """
    Performs backtesting on the given draws data.
    `draws_data` is expected to be a list of (timestamp, mains, bonus1, bonus2, draw_type) tuples,
    sorted from most recent to oldest.
    """
    logging.debug(f"backtest_strategy received {len(draws_data)} draws for processing.")
    results = []

    if len(draws_data) < 2:
        logging.info("Not enough draws for backtesting (need at least 2). Skipping backtest.")
        return [], [], []

    # To simulate adaptive behavior in backtesting, we need to re-run the prediction logic
    # for each historical point, just as the live system would.

    # Store a list of (prediction, bonus_used_for_prediction) for repetition detection during backtest
    backtest_prediction_history = []

    for i in range(len(draws_data) - 1):
        # The 'future_draws' for building frequency map are those *older* than the current draw being predicted
        # and also older than the target draw.
        # So, for predicting draw `i+1`, we use draws from `i+2` onwards for frequency map.
        # And the bonus will be based on `draws_data[i+1]` (the one we are trying to predict).

        # This is the draw whose numbers we are trying to predict (the 'actual' draw for this backtest step)
        next_actual_draw_data = draws_data[i + 1]
        next_actual_draw_timestamp, next_actual_mains, next_actual_b1, next_actual_b2, next_actual_draw_type = next_actual_draw_data

        # This is the most recent draw *available at the time of prediction* for `next_actual_draw_data`
        # It's `draws_data[i]`, and its bonuses will be used to determine the prediction basis.
        current_draw_data_for_prediction = draws_data[i]
        current_draw_b1_for_prediction = current_draw_data_for_prediction[2] # bonus1 of the most recent draw available

        # The data used to build the frequency map should be all draws *older* than the current draw for prediction
        # i.e., draws_data from index i+1 onwards.
        draws_for_freq_map = draws_data[i+1:]

        if not draws_for_freq_map:
            logging.debug(f"Not enough historical data to build frequency map for backtest step {i}. Skipping.")
            continue

        freq_map = build_frequency_map(draws_for_freq_map)

        # Determine the bonus for prediction algo based on the historical context
        all_bonus_numbers_for_prediction_basis = [d[2] for d in draws_for_freq_map] + [d[3] for d in draws_for_freq_map]
        bonus_counts_for_prediction_basis = Counter(all_bonus_numbers_for_prediction_basis)
        sorted_bonuses_for_prediction_basis = [b for b, _ in bonus_counts_for_prediction_basis.most_common()]

        bonus_for_prediction_algo = None
        # Find the second most common bonus from older draws that is NOT the latest_actual_bonus1
        for b in sorted_bonuses_for_prediction_basis:
            if b != current_draw_b1_for_prediction:
                bonus_for_prediction_algo = b
                break

        # Fallback if no distinct second bonus is found from older draws
        if bonus_for_prediction_algo is None:
            bonus_for_prediction_algo = random.randint(1, 10)
            logging.warning(f"No distinct second bonus found from older draws for backtest prediction. Using random fallback: {bonus_for_prediction_algo}")

        # --- Adaptive Strategy Logic within Backtest ---
        # Simulate checking for repetition for the *current* prediction being made in backtest
        current_prediction_attempt_mains = predict_strategy(bonus_for_prediction_algo, freq_map)

        # Check if this prediction is a repeat of recent predictions in the backtest history
        is_repeating = False
        if len(backtest_prediction_history) >= REPETITION_THRESHOLD_BACKTEST:
            # Get the last REPETITION_THRESHOLD_BACKTEST predictions
            recent_predictions = [item['prediction'] for item in backtest_prediction_history[-REPETITION_THRESHOLD_BACKTEST:]]
            if all(p == current_prediction_attempt_mains for p in recent_predictions):
                is_repeating = True
                logging.info(f"Backtest: Detected repeating prediction {current_prediction_attempt_mains} for {next_actual_draw_timestamp}. Attempting adaptive strategy.")

        if is_repeating:
            # Try to find a bonus that led to 3 hits in *past* backtests (or a different logic)
            # For simplicity in backtest, let's just pick a random bonus from 1-10 that's not the current one
            # A more sophisticated approach would store and use actual '3-hit' bonuses from previous backtest runs.

            # Find a 'successful' bonus from the *actual* historical draws that produced 3 hits
            # This is a simplification; a real adaptive system would need to store which bonuses were 'good'
            # from *its own* past predictions, not just any historical draw.
            # For now, we'll just pick a random bonus different from the one that led to repetition.
            successful_bonuses_from_past_actual_draws = []
            # Look at draws that are *older* than the current prediction target
            for k in range(i + 2, len(draws_data)): # Look at draws older than `current_draw_data_for_prediction`
                past_draw = draws_data[k]
                # This is a placeholder for a more robust "successful bonus" retrieval
                # For a true adaptive backtest, you'd need to have a record of which bonuses, when used
                # in *your strategy*, led to 3 hits in the past.
                # For now, we'll just pick a random bonus different from the one that led to repetition.
                successful_bonuses_from_past_actual_draws.extend([past_draw[2], past_draw[3]])

            unique_successful_bonuses = list(set(successful_bonuses_from_past_actual_draws))
            if unique_successful_bonuses:
                # Pick a random successful bonus that is different from the current one
                new_bonus_options = [b for b in unique_successful_bonuses if b != bonus_for_prediction_algo]
                if new_bonus_options:
                    # Corrected: Use bonus_for_prediction_algo for both in logging
                    logging.debug(f"Backtest: Switching bonus for prediction from {bonus_for_prediction_algo} to {bonus_for_prediction_algo} due to repetition.")
                    bonus_for_prediction_algo = random.choice(new_bonus_options)
                    current_prediction_attempt_mains = predict_strategy(bonus_for_prediction_algo, freq_map) # Re-predict with new bonus
                else:
                    logging.warning("Backtest: No *different* successful bonuses found for adaptation. Sticking with current random fallback.")
            else:
                logging.warning("Backtest: No 'successful' bonuses found to adapt to. Sticking with current random fallback.")


        predicted_mains_for_next = current_prediction_attempt_mains # Use the potentially adapted prediction

        hits = len(set(predicted_mains_for_next).intersection(set(next_actual_mains)))

        result_entry = {
            'target_draw_time': next_actual_draw_timestamp,
            'draw_date': next_actual_draw_timestamp.strftime('%Y-%m-%d'),
            'draw_type': next_actual_draw_type,
            'strategy_used': f'bonus_based_on_{bonus_for_prediction_algo}',
            'bonus': [current_draw_b1_for_prediction, draws_data[i][3]], # The bonuses of the draw used for prediction basis
            'prediction': predicted_mains_for_next,
            'actual_mains': next_actual_mains,
            'actual_bonuses': [next_actual_b1, next_actual_b2],
            'hits': hits
        }
        results.append(result_entry)
        backtest_prediction_history.append({'prediction': predicted_mains_for_next, 'hits': hits}) # Store for future repetition check
        logging.debug(f"Backtest result for {next_actual_draw_timestamp.strftime('%Y-%m-%d %H:%M')}: Predicted {predicted_mains_for_next}, Actual {next_actual_mains}, Hits {hits}.")

    all_main_numbers = []
    for _, mains, _, _, _ in draws_data:
        all_main_numbers.extend(mains)
    main_ranking = Counter(all_main_numbers).most_common()

    all_bonus_numbers = []
    for _, _, b1, b2, _ in draws_data:
        all_bonus_numbers.append(b1)
        all_bonus_numbers.append(b2)
    bonus_ranking = Counter(all_bonus_numbers).most_common()

    logging.info(f"Backtesting completed. Generated {len(results)} results.")
    return results, main_ranking, bonus_ranking

# --- Data Class for Prediction Results (for cleaner data passing) ---
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
        return {
            'target_draw_time': self.target_draw_time,
            'strategy_used': self.strategy_used,
            'bonus': self.bonus,
            'prediction': self.prediction,
            'actual_mains': self.actual_mains,
            'actual_bonuses': self.actual_bonuses,
            'hits': self.hits
        }

def get_next_target_draw_time(current_time_harare):
    """
    Calculates the next official Gosloto 5/50 draw time (08:30 or 20:30 CAT).
    """
    now_harare = current_time_harare.astimezone(harare_tz).replace(second=0, microsecond=0)

    today_morning_draw = now_harare.replace(hour=8, minute=30, second=0, microsecond=0)
    today_evening_draw = now_harare.replace(hour=20, minute=30, second=0, microsecond=0)

    if now_harare < today_morning_draw:
        next_target = today_morning_draw
    elif now_harare < today_evening_draw:
        next_target = today_evening_draw
    else:
        next_target = (now_harare + timedelta(days=1)).replace(hour=8, minute=30, second=0, microsecond=0)

    logging.debug(f"Calculated next target draw time: {next_target.strftime('%Y-%m-%d %H:%M %Z')}")
    return next_target

# --- Flask Routes ---

@app.context_processor
def inject_global_data():
    """Injects global data into all templates."""
    logged_in = 'firebase_uid' in session
    is_subscribed = False
    is_admin = session.get('is_admin', False)

    firebase_uid = session.get('firebase_uid')
    if firebase_uid and db:
        try:
            user_doc = get_user_doc_ref(firebase_uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                is_subscribed = user_data.get('is_subscribed', False)
        except Exception as e:
            logging.error(f"Error fetching user subscription status for {firebase_uid}: {e}")
            is_subscribed = False

    banner_message = "Welcome to Hilzhosting 5/50! Log in or Register to get started."
    if db:
        try:
            banner_doc = get_public_banner_doc_ref().get()
            if banner_doc.exists:
                banner_message = banner_doc.to_dict().get('message', banner_message)
        except Exception as e:
            logging.error(f"Error fetching banner message: {e}")

    return {
        'logged_in': logged_in,
        'is_subscribed': is_subscribed,
        'is_admin': is_admin,
        'banner_message': banner_message,
        'now': datetime.now(harare_tz),
        'firebase_web_client_config_json': FIREBASE_WEB_CLIENT_CONFIG_JSON
    }


@app.route('/')
def index():
    """Renders the main index page with latest draw and prediction data."""
    logging.info("--- Rendering Index Page ---") # DEBUG
    if db is None:
        flash("Database not initialized. Please check server logs.", "error")
        return render_template('index.html', error="Database connection error.")

    latest_draw_time, latest_actual_mains, latest_actual_bonus1, latest_actual_bonus2, _ = get_latest_actual_draw()
    latest_actual_bonuses = [latest_actual_bonus1, latest_actual_bonus2] if latest_actual_bonus1 is not None else []
    logging.debug(f"INDEX: Latest actual draw for display: {latest_draw_time}") # DEBUG

    current_prediction_vs_actual = None
    generated_main_combos_3 = []
    generated_main_combos_2 = []
    live_prediction_history = []
    live_performance_stats = {} # MODIFIED: Initialize stats dictionary

    # Define the history window for the index page prediction
    HISTORY_WINDOW_DAYS_INDEX = 60

    # --- Fetch public prediction first ---
    public_prediction_doc = get_public_prediction_doc_ref().get()
    if public_prediction_doc.exists:
        pred_data = public_prediction_doc.to_dict()
        pred_time = normalize_to_harare_time(pred_data.get('target_draw_time')) # Use helper function

        current_prediction_vs_actual = PredictionResult(
            target_draw_time=pred_time,
            strategy_used=pred_data.get('strategy_used', 'N/A'),
            bonus=pred_data.get('bonus', []),
            prediction=pred_data.get('prediction', []),
            actual_mains=pred_data.get('actual_mains', []),
            actual_bonuses=pred_data.get('actual_bonuses', []),
            hits=pred_data.get('hits')
        )
        logging.debug(f"INDEX: Fetched public prediction for display targeting: {current_prediction_vs_actual.target_draw_time}") # DEBUG
    else:
        logging.warning("INDEX: No public prediction found. A temporary one may be generated if needed.")
        # Fallback to generating a temporary prediction for display if public one doesn't exist
        # Use the defined history window for this temporary generation
        historical_draws = get_historical_draws_from_firestore(history_window_days=HISTORY_WINDOW_DAYS_INDEX)
        if historical_draws:
            # Replicate the backtest logic for temporary display prediction
            if len(historical_draws) >= 2:
                # `current_draw_b1` for this context is `historical_draws[0][2]` (latest actual bonus1)
                # `future_draws` for this context is `historical_draws[1:]` (all but the latest)
                current_draw_b1_for_temp_pred = historical_draws[0][2]
                draws_for_freq_map = historical_draws[1:]
            else: # Not enough data for backtest-like derivation, fallback to simpler
                current_draw_b1_for_temp_pred = latest_actual_bonus1
                draws_for_freq_map = historical_draws # Use all available for freq map

            freq_map = build_frequency_map(draws_for_freq_map)

            # Determine bonus for prediction algo based on backtest logic
            all_bonus_numbers_for_temp_pred = [d[2] for d in draws_for_freq_map] + [d[3] for d in draws_for_freq_map]
            bonus_counts_for_temp_pred = Counter(all_bonus_numbers_for_temp_pred)
            sorted_bonuses_for_temp_pred = [b for b, _ in bonus_counts_for_temp_pred.most_common()]

            bonus_for_prediction_algo = None
            for b in sorted_bonuses_for_temp_pred:
                if b != current_draw_b1_for_temp_pred:
                    bonus_for_prediction_algo = b
                    break

            if bonus_for_prediction_algo is None:
                bonus_for_prediction_algo = random.randint(1, 10)
                logging.warning("No distinct second bonus found for temporary prediction. Using random fallback.")

            # The bonuses stored with the prediction should still be the actual latest bonuses
            bonus_for_storage_primary = latest_actual_bonus1
            bonus_for_storage_secondary = latest_actual_bonus2
            if bonus_for_storage_primary is None:
                bonus_for_storage_primary = random.randint(1, 10)
            if bonus_for_storage_secondary is None:
                temp_rand_bonus = random.randint(1, 10)
                if temp_rand_bonus == bonus_for_storage_primary and bonus_for_storage_primary < 10:
                    bonus_for_storage_secondary = temp_rand_bonus + 1
                elif temp_rand_bonus == bonus_for_storage_primary and bonus_for_storage_primary > 1:
                    bonus_for_storage_secondary = temp_rand_bonus - 1
                else:
                    bonus_for_storage_secondary = temp_rand_bonus

            predicted_mains = predict_strategy(bonus_for_prediction_algo, freq_map)

            current_prediction_vs_actual = PredictionResult(
                target_draw_time=get_next_target_draw_time(datetime.now(harare_tz)),
                strategy_used="Smart Hybrid (Temporary - Backtest Logic)",
                bonus=[bonus_for_storage_primary, bonus_for_storage_secondary],
                prediction=predicted_mains,
                actual_mains=None,
                actual_bonuses=None,
                hits=None
            )
            logging.debug(f"INDEX: Generated temporary prediction for display: {current_prediction_vs_actual.target_draw_time}") # DEBUG
        else:
            logging.warning("INDEX: Not enough historical data to generate even a temporary prediction for display.")

    if current_prediction_vs_actual and current_prediction_vs_actual.prediction:
        generated_main_combos_3 = list(itertools.combinations(current_prediction_vs_actual.prediction, 3))
        generated_main_combos_2 = list(itertools.combinations(current_prediction_vs_actual.prediction, 2))

    # Fetch live prediction history for subscribed users only
    if session.get('logged_in') and session.get('is_subscribed'):
        user_uid = session['firebase_uid']
        logging.info(f"INDEX: Fetching live prediction history for subscribed user: {user_uid}") # DEBUG
        raw_history = []
        try:
            history_docs = get_user_predictions_history_ref(user_uid).order_by('target_draw_time', direction=firestore.Query.DESCENDING).limit(10).stream()
            for doc in history_docs:
                data = doc.to_dict()
                pred_time = normalize_to_harare_time(data.get('target_draw_time')) # Use helper function

                history_item = PredictionResult(
                    target_draw_time=pred_time,
                    strategy_used=data.get('strategy_used', 'N/A'),
                    bonus=data.get('bonus', []),
                    prediction=data.get('prediction', []),
                    actual_mains=data.get('actual_mains', []),
                    actual_bonuses=data.get('actual_bonuses', []),
                    hits=data.get('hits')
                )
                raw_history.append(history_item)
                # --- ADDED DETAILED LOG FOR EACH HISTORY ITEM ---
                logging.debug(f"INDEX: History item loaded - Target: {history_item.target_draw_time}, "
                              f"Prediction: {history_item.prediction}, "
                              f"Actual Mains: {history_item.actual_mains}, "
                              f"Hits: {history_item.hits}")
            live_prediction_history = raw_history
            logging.info(f"INDEX: Successfully loaded {len(live_prediction_history)} items into live history for user {user_uid}.") # DEBUG

            # START MODIFIED: Calculate Live Performance Stats
            if live_prediction_history:
                # Filter for predictions that have been checked against actual results
                completed_predictions = [p for p in live_prediction_history if p.actual_mains]
                total_completed = len(completed_predictions)

                if total_completed > 0:
                    hits_3_or_more = sum(1 for p in completed_predictions if p.hits is not None and p.hits >= 3)
                    hits_2_or_more = sum(1 for p in completed_predictions if p.hits is not None and p.hits >= 2)
                    hits_1_or_more = sum(1 for p in completed_predictions if p.hits is not None and p.hits >= 1)
                    strike_rate = (hits_1_or_more / total_completed) * 100

                    live_performance_stats = {
                        'total_analyzed': total_completed,
                        'strike_rate': f"{strike_rate:.1f}",
                        'hits_2_plus': hits_2_or_more,
                        'hits_3_plus': hits_3_or_more
                    }
            # END MODIFIED
        except Exception as e:
            logging.error(f"INDEX: Error fetching live prediction history for user {user_uid}: {e}")
            live_prediction_history = []


    return render_template('index.html',
                           latest_draw_time=latest_draw_time,
                           latest_actual_mains=latest_actual_mains,
                           latest_actual_bonuses=latest_actual_bonuses,
                           current_prediction_vs_actual=current_prediction_vs_actual,
                           generated_main_combos_3=generated_main_combos_3,
                           generated_main_combos_2=generated_main_combos_2,
                           live_prediction_history=live_prediction_history,
                           live_performance_stats=live_performance_stats) # MODIFIED: Pass stats to template


@app.route('/backtest_results', methods=['GET', 'POST']) # Added POST method
@login_required
@subscription_required
#@admin_required # Commented out as per previous conversation if not strictly needed
def backtest_results_page():
    """Renders the backtest results page with optional date filtering."""
    if db is None:
        flash("Database not initialized. Please check server logs.", "error")
        return render_template('backtest_results.html', error="Database connection error.")

    start_date = request.form.get('start_date') if request.method == 'POST' else None
    end_date = request.form.get('end_date') if request.method == 'POST' else None

    # Fetch historical draws with optional date filtering
    # For backtesting, we might still want to fetch all available data or a large window
    historical_draws = get_historical_draws_from_firestore(start_date=start_date, end_date=end_date)

    backtest_results = []
    main_ranking = []
    bonus_ranking = []

    if historical_draws:
        backtest_results, main_ranking, bonus_ranking = backtest_strategy(historical_draws)

    total_backtests = len(backtest_results)
    total_hits = sum(r['hits'] for r in backtest_results)

    # Calculate total predicted numbers for average hit rate
    total_predicted_numbers = 0
    for result in backtest_results:
        # Assuming each prediction is always 4 numbers as per predict_strategy
        total_predicted_numbers += len(result.get('prediction', []))

    average_hit_rate = 0
    if total_predicted_numbers > 0:
        average_hit_rate = (total_hits / total_predicted_numbers) * 100

    # Modified hit distribution calculations to be inclusive
    total_4_hits = sum(1 for r in backtest_results if r['hits'] == 4)
    total_3_hits = sum(1 for r in backtest_results if r['hits'] >= 3) # Includes 4 hits
    total_2_hits = sum(1 for r in backtest_results if r['hits'] >= 2) # Includes 3 and 4 hits
    total_1_hits = sum(1 for r in backtest_results if r['hits'] >= 1) # Includes 2, 3, and 4 hits
    total_0_hits = sum(1 for r in backtest_results if r['hits'] == 0) # Remains exclusive for 0 hits

    return render_template('backtest_results.html',
                           backtest_results=backtest_results,
                           total_backtests=total_backtests,
                           average_hit_rate=average_hit_rate,
                           total_4_hits=total_4_hits,
                           total_3_hits=total_3_hits,
                           total_2_hits=total_2_hits,
                           total_1_hits=total_1_hits,
                           total_0_hits=total_0_hits,
                           main_ranking=main_ranking,
                           bonus_ranking=bonus_ranking,
                           selected_start_date=start_date, # Pass selected dates back to template
                           selected_end_date=end_date)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        if not auth:
            logging.error("Firebase Auth client is not initialized.")
            return jsonify(success=False, message="Server error: Authentication service not available."), 500

        id_token = request.json.get('idToken')
        if not id_token:
            return jsonify(success=False, message="ID token missing."), 400

        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']
            is_admin_claim = decoded_token.get('admin', False)

            session['firebase_uid'] = uid
            session['logged_in'] = True
            session['is_admin'] = is_admin_claim

            user_doc = get_user_doc_ref(uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                session['is_subscribed'] = user_data.get('is_subscribed', False)
            else:
                get_user_doc_ref(uid).set({'email': decoded_token.get('email'), 'is_subscribed': False, 'telegram_chat_id': None}, merge=True)
                session['is_subscribed'] = False

            logging.info(f"User {uid} logged in. Admin: {is_admin_claim}, Subscribed: {session['is_subscribed']}")
            flash("Login successful!", "success")
            return jsonify(success=True, redirect=url_for('index'))

        except Exception as e:
            logging.error(f"Firebase ID token verification failed: {e}")
            return jsonify(success=False, message=f"Authentication failed: {e}"), 401

    # Pass firebase_web_client_config_json to the template
    return render_template('login.html', firebase_web_client_config_json=FIREBASE_WEB_CLIENT_CONFIG_JSON)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if request.method == 'POST':
        if not auth:
            logging.error("Firebase Auth client is not initialized.")
            return jsonify(success=False, message="Server error: Authentication service not available."), 500

        id_token = request.json.get('idToken')
        email = request.json.get('email')
        if not id_token or not email:
            return jsonify(success=False, message="ID token or email missing."), 400

        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token['uid']

            user_doc_ref = get_user_doc_ref(uid)
            user_doc_ref.set({
                'email': email,
                'is_subscribed': False,
                'telegram_chat_id': None,
                'registered_at': firestore.SERVER_TIMESTAMP
            })

            if email == INITIAL_ADMIN_EMAIL:
                users_count = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users').stream()
                if sum(1 for _ in users_count) == 1:
                    auth.set_custom_user_claims(uid, {'admin': True})
                    logging.info(f"Set admin claim for initial admin user: {email}")
                    session['is_admin'] = True
                else:
                    logging.info(f"User {email} registered, but not setting admin claim as it's not the initial user.")
            else:
                session['is_admin'] = False

            session['firebase_uid'] = uid
            session['logged_in'] = True
            session['is_subscribed'] = False

            logging.info(f"User {uid} registered and logged in. Email: {email}")
            flash("Registration successful! Welcome to HBet 5/50.", "success")
            return jsonify(success=True, redirect=url_for('index'))

        except Exception as e:
            logging.error(f"Firebase registration or token verification failed: {e}")
            return jsonify(success=False, message=f"Registration failed: {e}"), 400

    # Pass firebase_web_client_config_json to the template
    return render_template('register.html', firebase_web_client_config_json=FIREBASE_WEB_CLIENT_CONFIG_JSON)

@app.route('/logout')
def logout():
    """Handles user logout."""
    session.clear()
    flash("You have been logged out.", "info")
    logging.info("User logged out.")
    return redirect(url_for('index'))

@app.route('/subscribe_info')
@login_required
def subscribe_info():
    """Renders the subscription information page."""
    return render_template('subscribe_info.html')

@app.route('/telegram_settings', methods=['GET', 'POST'])
@login_required
@subscription_required
def telegram_settings():
    """Handles Telegram settings for the logged-in user."""
    user_uid = session['firebase_uid']
    user_doc_ref = get_user_doc_ref(user_uid)
    user_data = user_doc_ref.get().to_dict() if user_doc_ref.get().exists else {}
    current_chat_id = user_data.get('telegram_chat_id', '')

    if request.method == 'POST':
        new_chat_id = request.form.get('telegram_chat_id', '').strip()
        if new_chat_id:
            try:
                user_doc_ref.update({'telegram_chat_id': new_chat_id})
                flash("Telegram Chat ID updated successfully!", "success")
                logging.info(f"User {user_uid} updated Telegram Chat ID to {new_chat_id}.")
            except Exception as e:
                flash(f"Error updating Telegram Chat ID: {e}", "error")
                logging.error(f"Error updating Telegram Chat ID for {user_uid}: {e}")
        else:
            flash("Telegram Chat ID cannot be empty.", "error")
        return redirect(url_for('telegram_settings'))

    return render_template('telegram_settings.html', current_chat_id=current_chat_id)

@app.route('/send_alert', methods=['POST'])
@login_required
@subscription_required
def send_alert():
    """Sends the current prediction as a Telegram alert to the subscribed user."""
    user_uid = session['firebase_uid']
    user_doc = get_user_doc_ref(user_uid).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    telegram_chat_id = user_data.get('telegram_chat_id')

    if not telegram_chat_id:
        flash("Your Telegram Chat ID is not set. Please go to Telegram Settings to set it up.", "error")
        return redirect(url_for('telegram_settings'))

    # --- Fetch prediction from public current prediction for alert ---
    public_prediction_doc = get_public_prediction_doc_ref().get()
    if not public_prediction_doc.exists:
        flash("No current public prediction available to send. Please wait for the next scheduled update.", "error")
        logging.warning(f"User {user_uid} attempted to send alert, but no public prediction found.")
        return redirect(url_for('index'))

    pred_data = public_prediction_doc.to_dict()
    predicted_mains = pred_data.get('prediction', [])
    selected_bonus_for_prediction = pred_data.get('bonus', [])[0] if pred_data.get('bonus') else None
    second_bonus = pred_data.get('bonus', [])[1] if pred_data.get('bonus') and len(pred_data['bonus']) > 1 else None

    if not predicted_mains or selected_bonus_for_prediction is None or second_bonus is None:
        flash("Incomplete prediction data in public record. Cannot send alert.", "error")
        logging.error(f"Incomplete public prediction data for alert: {pred_data}")
        return redirect(url_for('index'))

    prediction_message = (
        f"粕 Hilzhosting 5/50 Prediction 粕\n\n"
        f"Main Numbers: <b>{', '.join(map(str, predicted_mains))}</b>\n"
        f"Bonus Balls: <b>{selected_bonus_for_prediction}, {second_bonus}</b>\n\n"
        f"Good luck! "
    )

    success, message = send_telegram_message(prediction_message, telegram_chat_id)
    if success:
        flash(message, "success")
        logging.info(f"Prediction alert sent to user {user_uid} using public prediction.")

        # When sending an alert, we also record this prediction in the user's history
        # Ensure it's the same prediction as the public one
        target_draw_time_from_pred = normalize_to_harare_time(pred_data.get('target_draw_time')) # Use helper function

        prediction_doc_id = target_draw_time_from_pred.strftime('%Y-%m-%d_%H%M')
        user_predictions_history_ref = get_user_predictions_history_ref(user_uid)

        try:
            # Use the data directly from the public prediction
            user_predictions_history_ref.document(prediction_doc_id).set({
                'target_draw_time': target_draw_time_from_pred,
                'strategy_used': pred_data.get('strategy_used', 'Smart Hybrid'),
                'bonus': pred_data.get('bonus', []),
                'prediction': predicted_mains,
                'actual_mains': [], # Initialize as empty, to be updated later
                'actual_bonuses': [], # Initialize as empty
                'hits': 0, # Initialize as 0
                'timestamp_generated': firestore.SERVER_TIMESTAMP
            }, merge=True)
            logging.info(f"Public prediction saved/updated in history for user {user_uid} with ID {prediction_doc_id}.")
        except Exception as e:
            logging.error(f"Failed to save/update public prediction in history for user {user_uid}: {e}")
            flash("Failed to save prediction to history.", "error")

    else:
        flash(message, "error")
        logging.error(f"Failed to send prediction alert to user {user_uid}: {e}")

    return redirect(url_for('index'))

@app.route('/admin_panel', endpoint='admin_panel') # New route for the admin panel
@login_required
@admin_required
def admin_panel():
    """Renders the admin panel page."""
    return render_template('admin_panel.html')

@app.route('/admin_users', endpoint='admin_users')
@login_required
@admin_required
def admin_users():
    """Admin page to list all users and their details."""
    if db is None:
        flash("Database not initialized. Please check server logs.", "error")
        return render_template('admin_users.html', users=[], error="Database connection error.")

    users_list = []
    try:
        users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
        docs = users_ref.stream()
        for doc in docs:
            user_data = doc.to_dict()
            uid = doc.id

            is_admin_user = False
            try:
                user_record = auth.get_user(uid)
                if user_record.custom_claims and user_record.custom_claims.get('admin'):
                    is_admin_user = True
            except Exception as e:
                logging.warning(f"Could not fetch custom claims for user {uid}: {e}")

            users_list.append({
                'uid': uid,
                'email': user_data.get('email', 'N/A'),
                'is_subscribed': user_data.get('is_subscribed', False),
                'telegram_chat_id': user_data.get('telegram_chat_id', 'N/A'),
                'is_admin': is_admin_user,
                'created_at': user_data.get('registered_at') # Added created_at
            })
        logging.info(f"Fetched {len(users_list)} users for admin panel.")
    except Exception as e:
        flash(f"Error fetching users: {e}", "error")
        logging.error(f"Error fetching users for admin panel: {e}")
        users_list = []

    return render_template('admin_users.html', users=users_list)

@app.route('/admin_toggle_subscription/<uid>', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_toggle_subscription(uid):
    """Admin action to toggle user subscription status."""
    if db is None:
        flash("Database not initialized.", "error")
        return redirect(url_for('admin_users'))

    try:
        user_doc_ref = get_user_doc_ref(uid)
        user_doc = user_doc_ref.get()
        if user_doc.exists:
            current_status = user_doc.to_dict().get('is_subscribed', False)
            new_status = not current_status
            user_doc_ref.update({'is_subscribed': new_status})
            flash(f"Subscription status for {uid} toggled to {new_status}.", "success")
            logging.info(f"Admin {session['firebase_uid']} toggled subscription for {uid} to {new_status}.")
        else:
            flash("User not found.", "error")
            logging.warning(f"Admin {session['firebase_uid']} tried to toggle subscription for non-existent user {uid}.")
    except Exception as e:
        flash(f"Error toggling subscription: {e}", "error")
        logging.error(f"Error toggling subscription for {uid}: {e}")

    return redirect(url_for('admin_users'))

@app.route('/admin_banner_settings', endpoint='admin_banner_settings', methods=['GET', 'POST'])   # Explicitly setting endpoint
@login_required
@admin_required
def admin_banner_settings():
    """Admin page to manage the moving banner message."""
    if db is None:
        flash("Database not initialized.", "error")
        return render_template('admin_banner_settings.html', current_message="Database error.")

    banner_doc_ref = get_public_banner_doc_ref()
    current_message = "Welcome to Hilzhosting 5/50! Log in or Register to get started."

    if request.method == 'POST':
        new_message = request.form.get('banner_message', '').strip()
        if new_message:
            try:
                banner_doc_ref.set({'message': new_message}, merge=True)
                flash("Banner message updated successfully!", "success")
                logging.info(f"Admin {session['firebase_uid']} updated banner message to: {new_message[:50]}...")
            except Exception as e:
                flash(f"Error updating banner message: {e}", "error")
                logging.error(f"Error updating banner message: {e}")
        else:
            flash("Banner message cannot be empty.", "error")
        return redirect(url_for('admin_banner_settings'))
    else:
        try:
            banner_doc = banner_doc_ref.get()
            if banner_doc.exists:
                current_message = banner_doc.to_dict().get('message', current_message)
        except Exception as e:
            flash(f"Error fetching current banner message: {e}", "error")
            logging.error(f"Error fetching current banner message: {e}")

    return render_template('admin_banner_settings.html', current_message=current_message)

@app.route('/set_admin_claim', endpoint='set_admin_claim', methods=['GET', 'POST'])
@login_required
@admin_required
def set_admin_claim():
    """Admin page to grant or revoke admin claims for users."""
    if request.method == 'POST':
        email = request.form.get('email')
        action = request.form.get('action')

        if not email:
            flash("Email is required.", "error")
            return redirect(url_for('set_admin_claim'))

        try:
            user = auth.get_user_by_email(email)
            if action == 'grant':
                auth.set_custom_user_claims(user.uid, {'admin': True})
                flash(f"Admin claim granted to {email}.", "success")
                logging.info(f"Admin {session['firebase_uid']} granted admin claim to {email}.")
            elif action == 'revoke':
                auth.set_custom_user_claims(user.uid, {'admin': False})
                flash(f"Admin claim revoked from {email}.", "success")
                logging.info(f"Admin {session['firebase_uid']} revoked admin claim from {email}.")
            else:
                flash("Invalid action.", "error")
        except auth.UserNotFoundError:
            flash(f"User with email {email} not found.", "error")
            logging.warning(f"Admin {session['firebase_uid']} attempted to modify admin claim for non-existent user {email}.")
        except Exception as e:
            flash(f"Error setting admin claim: {e}", "error")
            logging.error(f"Error setting admin claim for {email}: {e}")

        return render_template('admin_set_admin_claim.html')

    return render_template('admin_set_admin_claim.html')

@app.route('/admin_delete_user/<uid>', methods=['POST'], endpoint='admin_delete_user')
@login_required
@admin_required
def admin_delete_user(uid):
    """Admin action to delete a user."""
    # Ensure db and auth are initialized (they should be if your app starts correctly)
    global db, auth
    if auth is None:
        flash("Authentication service not initialized.", "error")
        return redirect(url_for('admin_users'))

    try:
        # Delete user from Firebase Authentication
        auth.delete_user(uid)
        logging.info(f"User {uid} deleted from Firebase Auth.")

        # Optionally, delete user document from Firestore (if it exists)
        user_doc_ref = get_user_doc_ref(uid) # Assuming get_user_doc_ref is defined
        if user_doc_ref.get().exists:
            user_doc_ref.delete()
            logging.info(f"User document for {uid} deleted from Firestore.")

        flash(f"User {uid} and associated data deleted successfully.", "success")
        logging.info(f"Admin {session['firebase_uid']} deleted user {uid}.")
    except Exception as e:
        flash(f"Error deleting user {uid}: {e}", "error")
        logging.error(f"Error deleting user {uid}: {e}")

    return redirect(url_for('admin_users'))


@app.route('/force_prediction_update', methods=['POST'], endpoint='force_prediction_update') # New route for force update
@login_required
@admin_required
def force_prediction_update():
    """Forces an immediate prediction update by triggering the scheduled job."""
    logging.info("Admin requested force prediction update.")
    try:
        scrape_and_process_draws_job()
        flash("Prediction update initiated successfully!", "success")
        logging.info("Forced prediction update completed.")
    except Exception as e:
        flash(f"Error forcing prediction update: {e}", "error")
        logging.error(f"Error forcing prediction update: {e}")
    return redirect(url_for('admin_panel'))


# --- Scheduler for periodic tasks ---
scheduler = BackgroundScheduler(daemon=True, timezone=harare_tz)

# Define a constant for the history window in days
HISTORY_WINDOW_DAYS = 60 # Use the last 60 days of data for predictions

# Constants for adaptive strategy
REPETITION_THRESHOLD_LIVE = 2 # Number of identical consecutive predictions to trigger adaptation
REPETITION_THRESHOLD_BACKTEST = 2 # Number of identical consecutive predictions to trigger adaptation in backtest
HISTORY_FOR_REPETITION_CHECK = 5 # How many past predictions to check for repetition


# Moved function definitions here to ensure they are defined before being used by the scheduler
def update_all_user_predictions_job():
    """
    Scheduled job to generate and save predictions for all subscribed users
    for the next upcoming draw. This job will now preferentially use the newly generated public prediction
    if available, or generate one if not.
    """
    logging.info("Running scheduled update_all_user_predictions_job...")
    if db is None:
        logging.error("Firestore DB client is not initialized. Cannot update user predictions.")
        return

    try:
        # Get the public prediction first
        public_prediction_doc = get_public_prediction_doc_ref().get()

        if public_prediction_doc.exists:
            pred_data = public_prediction_doc.to_dict()
            predicted_mains = pred_data.get('prediction', [])
            selected_bonus_for_prediction = pred_data.get('bonus', [])[0] if pred_data.get('bonus') else None
            second_bonus = pred_data.get('bonus', [])[1] if pred_data.get('bonus') and len(pred_data['bonus']) > 1 else None
            
            next_target_draw_time = normalize_to_harare_time(pred_data.get('target_draw_time')) # Use helper function
            
            strategy_used = pred_data.get('strategy_used', 'Smart Hybrid')
            logging.info("Using existing public prediction for user updates.")
        else:
            logging.warning("No public prediction found when updating user predictions. Generating a new one.")
            # Fallback to generate if no public prediction exists (should be rare if scrape_and_process runs first)
            # Use the defined history window for this generation
            historical_draws = get_historical_draws_from_firestore(history_window_days=HISTORY_WINDOW_DAYS)
            if not historical_draws or len(historical_draws) < 2: # Need at least 2 draws for this logic
                logging.warning("Not enough historical data (need at least 2 draws) to generate predictions like backtesting. Skipping user prediction update.")
                return

            # For live prediction, the "current_draw_data" for determining the bonus input
            # is the latest actual draw (historical_draws[0]).
            # The "future_draws" for building the frequency map is historical_draws[1:]

            latest_actual_bonus1 = historical_draws[0][2] # bonus1 of the most recent draw
            latest_actual_bonus2 = historical_draws[0][3] # bonus2 of the most recent draw
            draws_for_freq_map = historical_draws[1:] # All draws except the latest one

            freq_map = build_frequency_map(draws_for_freq_map)

            all_bonus_numbers_for_prediction_basis = [d[2] for d in draws_for_freq_map] + [d[3] for d in draws_for_freq_map]
            bonus_counts_for_prediction_basis = Counter(all_bonus_numbers_for_prediction_basis)
            sorted_bonuses_for_prediction_basis = [b for b, _ in bonus_counts_for_prediction_basis.most_common()]

            bonus_for_prediction_algo = None
            # Find the second most common bonus from older draws that is NOT the latest_actual_bonus1
            for b in sorted_bonuses_for_prediction_basis:
                if b != latest_actual_bonus1:
                    bonus_for_prediction_algo = b
                    break

            # Fallback if no distinct second bonus is found from older draws
            if bonus_for_prediction_algo is None:
                bonus_for_prediction_algo = random.randint(1, 10)
                logging.warning(f"No distinct second bonus found from older draws for prediction. Using random fallback: {bonus_for_prediction_algo}")

            next_target_draw_time = get_next_target_draw_time(datetime.now(harare_tz))
            predicted_mains = predict_strategy(bonus_for_prediction_algo, freq_map)

            # The bonuses stored with the prediction should still be the actual bonuses from the latest draw,
            # as these are the ones the user will relate to.
            selected_bonus_for_prediction = latest_actual_bonus1
            second_bonus = latest_actual_bonus2 # latest_actual_bonus2
            strategy_used = 'Smart Hybrid (Live - Backtest Logic)'


        # Fetch all users who are subscribed
        users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
        subscribed_users_query = users_ref.where(filter=FieldFilter('is_subscribed', '==', True))

        subscribed_users_docs = subscribed_users_query.stream()
        subscribed_uids = [doc.id for doc in subscribed_users_docs]
        logging.info(f"Found {len(subscribed_uids)} subscribed users to update predictions for.")

        if not subscribed_uids:
            logging.info("No subscribed users found. Skipping individual prediction saving.")
            return

        if not predicted_mains or selected_bonus_for_prediction is None or second_bonus is None or next_target_draw_time is None:
            logging.error("Incomplete prediction data after generation/fetch. Cannot save to user history.")
            return


        for user_uid in subscribed_uids:
            user_predictions_history_ref = get_user_predictions_history_ref(user_uid)
            prediction_doc_id = next_target_draw_time.strftime('%Y-%m-%d_%H%M')

            # Check if a prediction for this specific draw already exists
            existing_prediction_doc = user_predictions_history_ref.document(prediction_doc_id).get()
            if existing_prediction_doc.exists:
                logging.info(f"Prediction for {prediction_doc_id} for user {user_uid} already exists. Skipping overwrite.")
                continue # Skip to the next user if a prediction already exists for this draw time

            try:
                user_predictions_history_ref.document(prediction_doc_id).set({
                    'target_draw_time': next_target_draw_time,
                    'strategy_used': strategy_used,
                    'bonus': [selected_bonus_for_prediction, second_bonus], # Store both
                    'prediction': predicted_mains,
                    'actual_mains': [], # Initialize as empty, to be updated later
                    'actual_bonuses': [], # Initialize as empty
                    'hits': 0, # Initialize as 0
                    'timestamp_generated': firestore.SERVER_TIMESTAMP
                }, merge=True)
                logging.info(f"Generated and saved prediction for user {user_uid} for draw {prediction_doc_id}. Target time: {next_target_draw_time.strftime('%Y-%m-%d %H:%M %Z')}")
            except Exception as e:
                logging.error(f"Failed to save prediction for user {user_uid} for draw {prediction_doc_id}: {e}")

    except Exception as e:
        logging.error(f"Error in update_all_user_predictions_job: {e}")


def check_and_update_prediction_hits_job():
    """
    Scheduled job to check and update hits for past predictions against the latest actual draw.
    """
    # --- ADDED DEBUG LOGGING ---
    logging.info("--- JOB START: check_and_update_prediction_hits_job ---")
    if db is None:
        logging.error("HITS_JOB: Firestore DB client is not initialized. Cannot check prediction hits.")
        return

    latest_actual_draw_time, latest_actual_mains, latest_actual_bonus1, latest_actual_bonus2, _ = get_latest_actual_draw()

    if not latest_actual_draw_time or not latest_actual_mains:
        logging.info("HITS_JOB: No latest actual draw available to check hits against. Skipping job.")
        return

    # --- ADDED DEBUG LOGGING ---
    logging.info(f"HITS_JOB: Found latest actual draw. Time: {latest_actual_draw_time.strftime('%Y-%m-%d %H:%M:%S %Z')}, Mains: {latest_actual_mains}")

    # Adjust the latest actual draw time to match the expected target_draw_time format (08:30 or 20:30)
    target_draw_time_to_check = latest_actual_draw_time.replace(second=0, microsecond=0)

    if target_draw_time_to_check.hour < 12:
        target_draw_time_to_check = target_draw_time_to_check.replace(hour=8, minute=30)
    else:
        target_draw_time_to_check = target_draw_time_to_check.replace(hour=20, minute=30)

    # --- ADDED DEBUG LOGGING ---
    logging.info(f"HITS_JOB: Normalized target time to check against predictions: {target_draw_time_to_check.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    prediction_doc_id = target_draw_time_to_check.strftime('%Y-%m-%d_%H%M')
    logging.info(f"HITS_JOB: Generated prediction document ID to search for: '{prediction_doc_id}'")

    try:
        users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
        all_users_docs = users_ref.stream()
        
        user_list_for_check = list(all_users_docs) # Convert to list to avoid "stream consumed" issues
        if not user_list_for_check:
            logging.info("HITS_JOB: No users found in the database to check.")
            return
            
        logging.info(f"HITS_JOB: Starting to loop through {len(user_list_for_check)} user(s) to update hits...")

        for user_doc in user_list_for_check:
            user_uid = user_doc.id
            logging.debug(f"HITS_JOB: Checking user: {user_uid}")

            user_predictions_history_ref = get_user_predictions_history_ref(user_uid)
            prediction_doc_ref = user_predictions_history_ref.document(prediction_doc_id)
            prediction_doc = prediction_doc_ref.get()

            if prediction_doc.exists:
                pred_data = prediction_doc.to_dict()
                logging.debug(f"HITS_JOB: Found matching prediction doc '{prediction_doc_id}' for user {user_uid}.")
                
                if not pred_data.get('actual_mains'):
                    predicted_mains = pred_data.get('prediction', [])
                    hits = len(set(predicted_mains).intersection(set(latest_actual_mains)))

                    try:
                        prediction_doc_ref.update({
                            'actual_mains': latest_actual_mains,
                            'actual_bonuses': [latest_actual_bonus1, latest_actual_bonus2],
                            'hits': hits,
                            'timestamp_checked': firestore.SERVER_TIMESTAMP
                        })
                        # --- ADDED SUCCESS LOG ---
                        logging.info(f"HITS_JOB: SUCCESS! Updated hits for user {user_uid}, draw '{prediction_doc_id}'. Hits: {hits}.")
                    except Exception as e:
                        logging.error(f"HITS_JOB: FAILED to update Firestore for user {user_uid}, doc '{prediction_doc_id}': {e}")
                else:
                    # --- ADDED ALREADY-UPDATED LOG ---
                    logging.debug(f"HITS_JOB: Prediction '{prediction_doc_id}' for user {user_uid} already has actual results. Skipping update.")
            else:
                # --- THIS IS THE MOST IMPORTANT LOG FOR YOUR ISSUE ---
                logging.warning(f"HITS_JOB: NO MATCH! No prediction document found with ID '{prediction_doc_id}' for user {user_uid}. Cannot update hits.")

    except Exception as e:
        logging.error(f"HITS_JOB: An unexpected error occurred during the main loop: {e}")
    
    logging.info("--- JOB END: check_and_update_prediction_hits_job ---")


def precompute_successful_bonuses_job():
    """
    Scheduled job to run a full backtest, identify 'successful' bonuses (those leading to 4+ hits),
    and store them in a dedicated Firestore document for quick lookups by the live prediction job.
    """
    logging.info("--- JOB START: precompute_successful_bonuses_job ---")
    if db is None:
        logging.error("INSIGHTS_JOB: Firestore DB client is not initialized. Cannot run.")
        return

    # Use a large window of historical data to get meaningful insights.
    historical_draws = get_historical_draws_from_firestore(history_window_days=180)
    if len(historical_draws) < 20: # Ensure we have enough data to be meaningful
        logging.warning(f"INSIGHTS_JOB: Not enough historical data ({len(historical_draws)} draws) to generate insights. Skipping.")
        return

    # Run the backtest strategy to get performance data.
    backtest_results, _, _ = backtest_strategy(historical_draws)
    if not backtest_results:
        logging.warning("INSIGHTS_JOB: Backtest generated no results. Cannot compute successful bonuses.")
        return

    successful_bonuses = set()
    for result in backtest_results:
        # We consider a bonus successful if it led to 4 or more hits.
        if result.get('hits', 0) >= 4:
            strategy_str = result.get('strategy_used', '')
            # The bonus used for prediction is stored like 'bonus_based_on_8'. We parse it.
            if 'bonus_based_on_' in strategy_str:
                try:
                    bonus_num = int(strategy_str.split('_')[-1])
                    successful_bonuses.add(bonus_num)
                except (ValueError, IndexError):
                    logging.warning(f"INSIGHTS_JOB: Could not parse bonus from strategy string: '{strategy_str}'")

    if not successful_bonuses:
        logging.info("INSIGHTS_JOB: No bonuses met the '4+ hits' criteria in this backtest run.")
        return

    try:
        insights_doc_ref = get_strategy_insights_doc_ref()
        insights_doc_ref.set({
            'bonuses_with_4_hits': sorted(list(successful_bonuses)),
            'last_updated': firestore.SERVER_TIMESTAMP,
            'source_draws_count': len(historical_draws)
        })
        logging.info(f"INSIGHTS_JOB: Successfully pre-computed and stored {len(successful_bonuses)} successful bonuses: {sorted(list(successful_bonuses))}")
    except Exception as e:
        logging.error(f"INSIGHTS_JOB: Failed to store strategy insights in Firestore: {e}")

    logging.info("--- JOB END: precompute_successful_bonuses_job ---")



def get_subscribed_users_with_telegram():
    if db is None: return []
    try:
        users_ref = db.collection('artifacts').document(get_app_id_for_firestore()).collection('users')
        query = users_ref.where(filter=FieldFilter('is_subscribed', '==', True))
        return [{'uid': doc.id, 'chat_id': doc.to_dict()['telegram_chat_id']} for doc in query.stream() if doc.to_dict().get('telegram_chat_id')]
    except Exception as e:
        logging.error(f"Failed to get subscribed users: {e}")
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
    
    msg = (f"🔮 *Upcoming 5/50 Prediction* 🔮\n\n"
           f"For the *{target_time.strftime('%H:%M')}* draw:\n\n"
           f"Mains: <b>{', '.join(map(str, pred_data.get('prediction', [])))}</b>\n"
           f"Bonuses: <b>{', '.join(map(str, pred_data.get('bonus', [])))}</b>\n\n"
           f"✨Good luck! ✨")
           
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
            
            msg = (f"🎉 *Draw Results & Your Hits* 🎉\n\n"
                   f"Results for *{latest_draw_time.strftime('%Y-%m-%d %H:%M')}*:\n"
                   f"Mains: <b>{', '.join(map(str, latest_mains))}</b>\n"
                   f"Bonuses: <b>{b1}, {b2}</b>\n\n"
                   f"Your Prediction: {', '.join(map(str, user_pred))}\n"
                   f"Your Hits: ✅ <b>{hits_str}</b>\n\n")
            if hit_count >= 3: msg += f"🎯 *Congratulations! You got {hit_count} hits!* 🎯"
            elif hit_count > 0: msg += f"Good job on {hit_count} hit(s)!"
            else: msg += "Better luck next time! 🤞"
            
            send_telegram_message(msg, user['chat_id'])
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"Failed to process results for {user['uid']}: {e}")



def scrape_and_process_draws_job():
    """Job to scrape draws, store them, generate public prediction, update user predictions, and check hits."""
    logging.info("Running scheduled scrape and process draws job...")

    # 1. Scrape and Store Draws
    draws_data, error = fetch_draws_from_website()
    if error:
        logging.error(f"Scraping job failed: {error}")
        return
    inserted_count = store_draws_to_firestore(draws_data)
    logging.info(f"Scrape and store job finished. Inserted {inserted_count} new draws.")

    # --- Generate and Save Public Prediction ---
    # Use the defined history window for public prediction generation
    historical_draws = get_historical_draws_from_firestore(history_window_days=HISTORY_WINDOW_DAYS)
    logging.debug(f"DEBUG: Length of historical_draws before prediction check: {len(historical_draws)}") # Added debug log
    if not historical_draws or len(historical_draws) < 2: # Need at least 2 draws for this logic
        logging.warning("Not enough historical data (need at least 2 draws) to generate a public prediction like backtesting.")
        return # Cannot proceed with backtest-like prediction

    # For live prediction, the "current_draw_data" for determining the bonus input
    # is the latest actual draw (historical_draws[0]).
    # The "future_draws" for building the frequency map is historical_draws[1:]

    latest_actual_bonus1 = historical_draws[0][2] # bonus1 of the most recent draw
    latest_actual_bonus2 = historical_draws[0][3] # bonus2 of the most recent draw
    draws_for_freq_map = historical_draws[1:] # All draws except the latest one

    freq_map = build_frequency_map(draws_for_freq_map)

    all_bonus_numbers_for_prediction_basis = [d[2] for d in draws_for_freq_map] + [d[3] for d in draws_for_freq_map]
    bonus_counts_for_prediction_basis = Counter(all_bonus_numbers_for_prediction_basis)
    sorted_bonuses_for_prediction_basis = [b for b, _ in bonus_counts_for_prediction_basis.most_common()]

    bonus_for_prediction_algo = None
    # Find the second most common bonus from older draws that is NOT the latest_actual_bonus1
    for b in sorted_bonuses_for_prediction_basis:
        if b != latest_actual_bonus1:
            bonus_for_prediction_algo = b
            break

    # Fallback if no distinct second bonus is found from older draws
    if bonus_for_prediction_algo is None:
        bonus_for_prediction_algo = random.randint(1, 9)
        logging.warning(f"No distinct second bonus found from older draws for prediction. Using random fallback: {bonus_for_prediction_algo}")

    # --- Adaptive Strategy Logic for Live Prediction ---
    # Fetch recent public predictions to check for repetition
    recent_public_predictions = []
    try:
        public_history_docs = get_public_prediction_history_ref().order_by('timestamp_generated', direction=firestore.Query.DESCENDING).limit(HISTORY_FOR_REPETITION_CHECK).stream()
        for doc in public_history_docs:
            data = doc.to_dict()
            recent_public_predictions.append(data.get('prediction', []))
    except Exception as e:
        logging.error(f"Error fetching public prediction history for repetition check: {e}")
        recent_public_predictions = [] # Reset to empty if error

    current_prediction_attempt_mains = predict_strategy(bonus_for_prediction_algo, freq_map)

    is_repeating_live = False
    if len(recent_public_predictions) >= REPETITION_THRESHOLD_LIVE:
        # Check if the last REPETITION_THRESHOLD_LIVE predictions were identical to each other
        # and also if the *new* prediction would be identical to them.
        first_recent_prediction = recent_public_predictions[0]
        if all(p == first_recent_prediction for p in recent_public_predictions) and \
           current_prediction_attempt_mains == first_recent_prediction:
            is_repeating_live = True

    if is_repeating_live:
        strategy_used = 'Smart Hybrid' # Default strategy name
        predicted_mains = current_prediction_attempt_mains # Start with the original prediction

        logging.info("Live: Detected repeating prediction. Attempting adaptive strategy using pre-computed insights.")
        try:
            # Fetch the pre-computed list of successful bonuses
            insights_doc = get_strategy_insights_doc_ref().get()
            if insights_doc.exists:
                successful_bonuses_for_adaptation = insights_doc.to_dict().get('bonuses_with_4_hits', [])

                if successful_bonuses_for_adaptation:
                    # Pick a random successful bonus that is different from the one that caused the repetition
                    new_bonus_options = [b for b in successful_bonuses_for_adaptation if b != bonus_for_prediction_algo]

                    if new_bonus_options:
                        new_bonus = random.choice(new_bonus_options)
                        logging.info(f"Live: Adapted bonus for prediction from {bonus_for_prediction_algo} to {new_bonus} due to repetition.")
                        # Re-predict with the new, successful bonus
                        predicted_mains = predict_strategy(new_bonus, freq_map)
                        strategy_used = 'Smart Hybrid (Adaptive - Repetition Detected)'
                    else:
                        logging.warning("Live: No *different* pre-computed successful bonuses found for adaptation. Using original prediction.")
                        strategy_used = 'Smart Hybrid (Live - No Adaptation Possible)'
                else:
                    logging.warning("Live: No pre-computed successful bonuses found in Firestore. Using original prediction.")
                    strategy_used = 'Smart Hybrid (Live - No Adaptation Data)'
            else:
                logging.warning("Live: Strategy insights document does not exist. Cannot adapt. Using original prediction.")
                strategy_used = 'Smart Hybrid (Live - No Adaptation Data)'
        except Exception as e:
            logging.error(f"Live: Error during adaptive strategy execution: {e}. Using original prediction.")
            strategy_used = 'Smart Hybrid (Live - Adaptation Failed)'
    else:
        # This is the original logic for when no repetition is detected
        predicted_mains = current_prediction_attempt_mains
        strategy_used = 'Smart Hybrid'


    next_target_draw_time = get_next_target_draw_time(datetime.now(harare_tz)) # Ensure target time is always current

    try:
        public_prediction_doc_ref = get_public_prediction_doc_ref()
        public_prediction_doc_ref.set({
            'target_draw_time': next_target_draw_time,
            'strategy_used': strategy_used,
            'bonus': [latest_actual_bonus1, latest_actual_bonus2], # Store actual latest bonuses for user context
            'prediction': predicted_mains,
            'actual_mains': [], # Initialize as empty, to be updated later
            'actual_bonuses': [], # Initialize as empty
            'hits': 0, # Initialize as 0
            'timestamp_generated': firestore.SERVER_TIMESTAMP
        }, merge=True)
        logging.info(f"Generated and saved public prediction for draw {next_target_draw_time.strftime('%Y-%m-%d %H:%M %Z')}")

        # Also save to public prediction history for repetition detection
        public_prediction_history_ref = get_public_prediction_history_ref()
        public_prediction_history_ref.add({
            'prediction': predicted_mains,
            'target_draw_time': next_target_draw_time,
            'timestamp_generated': firestore.SERVER_TIMESTAMP
        })
        logging.info(f"Saved public prediction to history for repetition check.")

    except Exception as e:
        logging.error(f"Failed to save public prediction: {e}")


    # 2. Update all subscribed user predictions for the *next* upcoming draw
    # This function will now preferentially use the newly generated public prediction
    update_all_user_predictions_job()

    # 3. Check and update hits for *past* predictions against the latest actual draw
    check_and_update_prediction_hits_job()

# Schedule the main processing job to run at specific minutes after the hour for a window
# This will run at 08:35, 08:40, 08:45, 08:50, 08:55 CAT
# And at 20:35, 20:40, 20:45, 20:50, 20:55 CAT
scheduler.add_job(
    scrape_and_process_draws_job,
    'cron',
    hour='8,20',
    minute='40,45,50',
    id='scrape_and_process_job',
    replace_existing=True
) 

# Schedule the new pre-computation job to run daily
scheduler.add_job(
    precompute_successful_bonuses_job,
    'cron',
    hour='6', # Run at 3 AM every day
    minute='50',
    id='precompute_bonuses_job',
    replace_existing=True
)

# Schedule the new send_prediction_alerts_job to run daily
scheduler.add_job(
    send_prediction_alerts_job,
    'cron',
    hour='7,13', # Run at 7 AM every day
    minute='10',
    id='send_alerts',
    replace_existing=True
)

# Schedule the new send_results_and_hits_job run daily
scheduler.add_job(
    send_results_and_hits_job,
    'cron',
    hour='8,20', # Run at 7 AM every day
    minute='58',
    id='send_results',
    replace_existing=True
)

# Start the scheduler
scheduler.start()
logging.info("Scheduler started with 'scrape_and_process_job' and 'precompute_bonuses_job'.")

# Shut down the scheduler when the app exits
atexit.register(lambda: scheduler.shutdown())
logging.info("Registered scheduler shutdown on exit.")
