"""
Flask app template for interacting with external web endpoints.

Features:
- Config via environment variables
- requests.Session with retries and timeouts
- Example GET and POST routes that call external APIs
- Basic error handling and logging
- Optional simple in-memory caching
"""

import os
import logging
from functools import wraps
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, abort
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration via environment variables
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
EXTERNAL_API_BASE = os.getenv("EXTERNAL_API_BASE", "https://ubs-gcc-2025-trivia-9ef655e6161b.herokuapp.com/")
EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY")  # optional
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))  # seconds

# Setup logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(
    EXTERNAL_API_BASE=EXTERNAL_API_BASE,
    EXTERNAL_API_KEY=EXTERNAL_API_KEY,
    REQUEST_TIMEOUT=REQUEST_TIMEOUT,
)

# Create a requests session with retries
def create_session(retries=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        status=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = create_session()

# Optional simple in-memory cache (for demonstration; not for production)
_cache = {}
def cache(ttl_seconds=60):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            key = (f.__name__, args, frozenset(kwargs.items()))
            entry = _cache.get(key)
            if entry:
                value, expires = entry
                if datetime.utcnow() < expires:
                    logger.debug("Cache hit for %s", key)
                    return value
                else:
                    logger.debug("Cache expired for %s", key)
                    _cache.pop(key, None)
            result = f(*args, **kwargs)
            _cache[key] = (result, datetime.utcnow() + timedelta(seconds=ttl_seconds))
            return result
        return wrapped
    return decorator

# Helper to call external API with configured session, headers, timeout
def call_external(path, method="GET", params=None, json=None, headers=None, raise_for_status=True):
    url = f"{app.config['EXTERNAL_API_BASE'].rstrip('/')}/{path.lstrip('/')}"
    req_headers = headers.copy() if headers else {}
    api_key = app.config.get("EXTERNAL_API_KEY")
    if api_key:
        # Example of an API key header — adjust per API spec
        req_headers.setdefault("Authorization", f"Bearer {api_key}")
    try:
        logger.debug("Outgoing %s %s params=%s json=%s headers=%s", method, url, params, json, req_headers)
        resp = session.request(method, url, params=params, json=json, headers=req_headers, timeout=app.config['REQUEST_TIMEOUT'])
        if raise_for_status:
            resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.exception("Error calling external service %s %s", method, url)
        # Re-raise or wrap as needed
        raise

# Example route: proxy a GET request to an external endpoint
@app.route("/external/trivia", methods=['GET'])
def external_get_info():
    # Example: /external/get-info?item=42
    item = request.args.get("item")
    if not item:
        return jsonify({"error": "missing 'item' query parameter"}), 400

    try:
        resp = call_external(f"/items/{item}", method="GET", params={"include": "details"})
        return jsonify({
            "status": "ok",
            "data": resp.json(),
        })
    except requests.HTTPError as he:
        return jsonify({"error": "external API returned error", "details": str(he)}), 502
    except Exception as e:
        return jsonify({"error": "failed to reach external API", "details": str(e)}), 503

# Example route: POST JSON payload to external endpoint
@app.route("/external/create", methods=["POST"])
def external_create():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "invalid or missing JSON body"}), 400
    try:
        resp = call_external("/items", method="POST", json=payload)
        return jsonify({"status": "created", "external_response": resp.json()}), 201
    except requests.HTTPError as he:
        return jsonify({"error": "external API returned error", "details": str(he), "text": getattr(he.response, "text", None)}), 502
    except Exception as e:
        return jsonify({"error": "failed to reach external API", "details": str(e)}), 503

@app.route("/ready")
def ready():
    # Optionally attempt a lightweight external call to verify dependency
    base = app.config.get("EXTERNAL_API_BASE")
    if not base:
        return jsonify({"ready": False, "reason": "no EXTERNAL_API_BASE configured"}), 500
    try:
        # HEAD or GET to base
        resp = session.head(base, timeout=2)
        if resp.status_code < 400:
            return jsonify({"ready": True})
        else:
            return jsonify({"ready": False, "status": resp.status_code}), 503
    except Exception:
        return jsonify({"ready": False, "reason": "cannot reach external API"}), 503

if __name__ == "__main__":
    # For dev only. In production run with gunicorn/uvicorn + workers.
    app.run(host=HOST, port=PORT, debug=DEBUG)