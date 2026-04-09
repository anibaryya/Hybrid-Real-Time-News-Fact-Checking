import sys
import os
import re
import io
import logging
import secrets
import tempfile
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

try:
    from ml_model.model import analyze_news
except:
    def analyze_news(text):
        return "Model temporarily unavailable"

try:
    from backend import database as dbstore
except:
    dbstore = None

from flask import Flask, request, jsonify, send_from_directory, session, abort
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from werkzeug.security import generate_password_hash, check_password_hash
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

from flask import Flask, request, jsonify, send_from_directory, session, abort

app = Flask(__name__)


# Use a stable fallback so sessions survive backend restarts in local/dev runs.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "veritai-local-dev-secret-change-me"
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=24),
)
CORS(
    app,
    supports_credentials=True,
    resources={r"/api/*": {"origins": ["http://127.0.0.1:5000", "http://localhost:5000"]}},
)

MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 20

GUEST_TTL = timedelta(minutes=30)
MEMBER_TTL = timedelta(hours=24)


@app.before_request
def _enforce_session_ttl():
    """
    - Guest sessions: expire after 30 minutes of inactivity.
    - Member sessions (email OTP / Google): expire after 24 hours of inactivity, or manual logout.
    """
    ut = session.get("user_type")
    if not ut:
        return
    now = datetime.now(timezone.utc)
    last_seen_raw = session.get("last_seen")
    try:
        last_seen = datetime.fromisoformat(last_seen_raw) if last_seen_raw else None
    except Exception:
        last_seen = None
    ttl = MEMBER_TTL if ut == "member" else GUEST_TTL
    if last_seen and (now - last_seen) > ttl:
        session.clear()
        return
    session["last_seen"] = now.isoformat()
    # Keep cookie across browser restarts up to TTL windows.
    session.permanent = True


try:
    dbstore.init_db()
except Exception:
    logger.exception("Database init failed — set DATABASE_URL (PostgreSQL)")
    # Keep the service running in guest/demo mode even when PostgreSQL isn't available.
    # Routes that require DB will still fail if called, but /api/analyze-news can work.

# ── URL / article text ────────────────────────────────────────
def extract_text_from_url(url: str) -> str | None:
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["nav", "footer", "script", "style", "aside", "header"]):
            tag.decompose()
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        ]
        if not paragraphs:
            return None
        return " ".join(paragraphs)[:MAX_TEXT_LENGTH]
    except requests.exceptions.RequestException as e:
        logger.error("URL fetch error for %s: %s", url, e)
        return None
    except Exception as e:
        logger.error("URL extraction error: %s", e)
        return None


def extract_article_preview(url: str, max_chars: int = 4000) -> dict | None:
    """Title + lead paragraphs for in-app 'read more'."""
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        title = ""
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = (og.get("content") or "").strip()
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        desc_og = soup.find("meta", property="og:description")
        if desc_og and desc_og.get("content"):
            meta_desc = (desc_og.get("content") or "").strip()
        else:
            md = soup.find("meta", attrs={"name": "description"})
            meta_desc = (md.get("content") or "").strip() if md else ""

        for tag in soup(["nav", "footer", "script", "style", "aside", "header"]):
            tag.decompose()
        paragraphs = [
            p.get_text(separator=" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 35
        ]
        body = " ".join(paragraphs)[:max_chars] if paragraphs else ""
        if not body and meta_desc:
            body = meta_desc[:max_chars]
        if not body:
            return {"title": title or "Article", "excerpt": "", "url": url}
        return {"title": title or "Article", "excerpt": body, "url": url}
    except Exception as e:
        logger.warning("article preview failed for %s: %s", url, e)
        return None


# ── OTP / mail / SMS ───────────────────────────────────────────
def _send_email_otp(to_addr: str, code: str) -> bool:
    brevo_api_key = (os.getenv("BREVO_API_KEY") or "").strip()
    sender_email = (os.getenv("BREVO_SENDER_EMAIL") or "").strip()
    sender_name = (os.getenv("BREVO_SENDER_NAME") or "VeritAI").strip()
    if not brevo_api_key or not sender_email:
        logger.warning("Brevo not configured — OTP for %s: %s", to_addr, code)
        return False
    subject = "VeritAI — Your verification code"
    plain = (
        f"Your VeritAI verification code is: {code}\n\n"
        "This code expires in 10 minutes.\n"
        "If you did not request this, you can safely ignore this email."
    )
    html = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; background:#0a0a0b; padding:24px;">
      <div style="max-width:560px; margin:0 auto; background:#111113; border:1px solid rgba(255,255,255,0.08); border-radius:14px; overflow:hidden;">
        <div style="padding:18px 20px; border-bottom:1px solid rgba(255,255,255,0.08);">
          <div style="font-size:20px; font-weight:800; letter-spacing:-0.3px; color:#f0ede8;">
            Verit<span style="color:#e8c97d;">AI</span>
          </div>
          <div style="margin-top:6px; font-size:12px; color:#9b9794; letter-spacing:1px; text-transform:uppercase;">
            SIGN-IN CODE
          </div>
        </div>
        <div style="padding:20px;">
          <p style="margin:0 0 12px; color:#f0ede8; font-size:14px; line-height:1.6;">
            Use this one-time code to continue signing in.
          </p>
          <div style="margin:18px 0; padding:16px; border-radius:12px; background:rgba(232,201,125,0.10); border:1px solid rgba(232,201,125,0.25); text-align:center;">
            <div style="font-size:28px; font-weight:900; letter-spacing:6px; color:#e8c97d;">{code}</div>
          </div>
          <p style="margin:0; color:#9b9794; font-size:12px; line-height:1.6;">
            This code expires in 10 minutes. If you didn’t request it, ignore this email.
          </p>
        </div>
      </div>
    </div>
    """
    try:
        payload = {
            "sender": {"name": sender_name, "email": sender_email},
            "to": [{"email": to_addr}],
            "subject": subject,
            "textContent": plain,
            "htmlContent": html,
        }
        res = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "api-key": brevo_api_key,
            },
            json=payload,
            timeout=20,
        )
        if res.status_code >= 300:
            logger.warning("Brevo send failed for %s: %s %s", to_addr, res.status_code, res.text[:300])
            return False
        logger.info("OTP email sent via Brevo to %s", to_addr)
        return True
    except Exception:
        logger.exception("Brevo send failed for %s", to_addr)
        return False


def _normalize_phone(raw: str) -> str | None:
    """E.164 only (must include + and country code), for SMS providers."""
    s = (raw or "").strip()
    if not s.startswith("+"):
        return None
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10 or len(digits) > 15:
        return None
    return "+" + digits


# ── News helpers ───────────────────────────────────────────────
def _newsapi_top(
    category: str | None = None,
    q: str | None = None,
    page_size: int = 12,
    country: str = "us",
) -> list:
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return []
    try:
        if category:
            if category == "general":
                url = (
                    "https://newsapi.org/v2/top-headlines"
                    f"?country={requests.utils.quote(country)}&pageSize={page_size}&apiKey={key}"
                )
            else:
                url = (
                    "https://newsapi.org/v2/top-headlines"
                    f"?country={requests.utils.quote(country)}&category={requests.utils.quote(category)}"
                    f"&pageSize={page_size}&apiKey={key}"
                )
        else:
            qq = q or "news"
            url = (
                "https://newsapi.org/v2/everything"
                f"?q={requests.utils.quote(qq)}&sortBy=publishedAt"
                f"&pageSize={page_size}&apiKey={key}"
            )
        r = requests.get(url, timeout=8).json()
        out = []
        for a in r.get("articles", []) or []:
            if not a.get("title") or "[Removed]" in (a.get("title") or ""):
                continue
            out.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "description": (a.get("description") or "").strip(),
                "image": a.get("urlToImage") or "",
                "publishedAt": a.get("publishedAt") or "",
            })
        return out[: page_size]
    except Exception as e:
        logger.warning("NewsAPI headlines failed: %s", e)
        return []


def _newsapi_top_query_local(q: str, country: str, page_size: int = 12) -> list:
    """
    Use NewsAPI top-headlines with country + query, which is better for
    location-prioritized category-like discovery than /everything.
    """
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return []
    try:
        url = (
            "https://newsapi.org/v2/top-headlines"
            f"?country={requests.utils.quote(country)}"
            f"&q={requests.utils.quote(q)}"
            f"&pageSize={page_size}&apiKey={key}"
        )
        r = requests.get(url, timeout=8).json()
        out = []
        for a in r.get("articles", []) or []:
            if not a.get("title") or "[Removed]" in (a.get("title") or ""):
                continue
            out.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "description": (a.get("description") or "").strip(),
                "image": a.get("urlToImage") or "",
                "publishedAt": a.get("publishedAt") or "",
            })
        return out[:page_size]
    except Exception as e:
        logger.warning("NewsAPI local query failed: %s", e)
        return []


@lru_cache(maxsize=512)
def _guess_country_code_from_coords(lat: float, lon: float) -> str | None:
    """
    Reverse-geocode lat/lon to a 2-letter country code for NewsAPI.
    Falls back to None if the lookup fails.
    """
    try:
        # Nominatim returns `address.country_code` (typically ISO-3166 alpha-2).
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?format=jsonv2&lat={lat}&lon={lon}&zoom=3&addressdetails=1"
        )
        headers = {"User-Agent": "VeritAI-NewsFactChecker/1.0 (compatible; +https://example.com)"}
        r = requests.get(url, headers=headers, timeout=6)
        r.raise_for_status()
        data = r.json() or {}
        cc = (((data.get("address") or {}).get("country_code") or "")).lower().strip()
        if len(cc) == 2:
            return cc
    except Exception:
        return None
    return None


def _resolve_country_from_request(default_country: str = "us") -> str:
    """
    Resolve and persist country code for location-based recommendations.
    - Uses lat/lon when provided
    - Caches in session so one successful detection can be reused
    """
    country = default_country
    cached = (session.get("country_code") or "").strip().lower()
    if len(cached) == 2:
        country = cached

    lat_raw = request.args.get("lat")
    lon_raw = request.args.get("lon")
    if lat_raw and lon_raw:
        try:
            # Round before cached reverse geocode to reduce churn/rate-limit issues.
            lat = round(float(lat_raw), 2)
            lon = round(float(lon_raw), 2)
            guessed = _guess_country_code_from_coords(lat, lon)
            if guessed:
                country = guessed
                session["country_code"] = country
        except Exception:
            pass
    return country


def _merge_unique_by_url(*lists: list) -> list:
    seen: set[str] = set()
    out: list = []
    for lst in lists:
        for a in lst or []:
            u = (a.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(dict(a))
    return out


def _genre_news_bundle(genre: str, country: str) -> list:
    """
    Fetch genre news with resilient fallbacks.
    Order:
      1) top-headlines by category for user's country
      2) top-headlines by category for US (broader fallback)
      3) everything query by genre keywords
    """
    cat = genre if genre != "general" else "general"
    a1 = _newsapi_top(category=cat, page_size=8, country=country)
    if a1:
        return a1
    # local query fallback keeps location preference while category feed is sparse
    a2 = _newsapi_top_query_local(q=f"{genre}", country=country, page_size=8)
    if a2:
        return a2
    a3 = _newsapi_top(category=cat, page_size=8, country="us")
    if a3:
        return a3
    return _newsapi_top(q=f"{genre} news", page_size=8)


# ── API routes ────────────────────────────────────────────────
@app.route("/api/ping")
def ping():
    return jsonify({"ok": True, "service": "veritai-backend"})


@app.route("/api/config")
def public_config():
    return jsonify({"googleClientId": os.getenv("GOOGLE_CLIENT_ID") or ""})


@app.route("/api/health")
def health():
    keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "NEWS_API_KEY": bool(os.getenv("NEWS_API_KEY")),
        "GNEWS_API_KEY": bool(os.getenv("GNEWS_API_KEY")),
        "SERPER_API_KEY": bool(os.getenv("SERPER_API_KEY")),
    }
    return jsonify({"status": "ok", "keys": keys}), 200


@app.route("/api/article-preview", methods=["POST"])
def article_preview():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url required"}), 400
    prev = extract_article_preview(url)
    if not prev:
        return jsonify({"error": "Could not load article"}), 422
    return jsonify(prev)


@app.route("/api/analyze-news", methods=["POST"])
def analyze():
    try:
        text = (request.form.get("text") or "").strip()
        url = (request.form.get("url") or "").strip()

        if not text and url:
            text = extract_text_from_url(url)
            if not text:
                return jsonify({"error": "Could not extract text from the provided URL."}), 422

        if not text:
            return jsonify({"error": "Please provide news text or a valid URL."}), 400

        if len(text) < MIN_TEXT_LENGTH:
            return jsonify({"error": f"Input too short (minimum {MIN_TEXT_LENGTH} characters)."}), 400

        text = text[:MAX_TEXT_LENGTH]
        input_mode = (request.form.get("input_mode") or "text").strip()[:24] or "text"
        result = analyze_news(text)

        payload = {
            "final_label": result["label"],
            "confidence": result["confidence"],
            "certainty": result["certainty"],
            "reason": result.get("reason", ""),
            "articles": result.get("articles", []),
            "scores": result.get("scores", {}),
            "followup_question": result.get("followup_question"),
            "followup_yes_prompt": result.get("followup_yes_prompt"),
        }
        uid = session.get("user_id")
        if session.get("user_type") == "member" and uid:
            try:
                dbstore.save_search_history(
                    int(uid),
                    input_mode,
                    text[:800],
                    str(result["label"]),
                    float(result["confidence"]),
                    str(result["certainty"]),
                )
            except Exception:
                logger.exception("Could not save search history")
        return jsonify(payload)
    except Exception:
        logger.exception("Unhandled error in /api/analyze-news")
        return jsonify({"error": "Internal server error. Please try again."}), 500


@app.route("/api/transcribe-audio", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    f = request.files["audio"]
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return jsonify({"error": "Voice input requires GROQ_API_KEY on the server."}), 503
    suffix = os.path.splitext(f.filename or "")[1] or ".webm"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        f.save(tmp.name)
        tmp.close()
        from groq import Groq

        client = Groq(api_key=api_key)
        with open(tmp.name, "rb") as audio_file:
            data = audio_file.read()
        tr = client.audio.transcriptions.create(
            file=(os.path.basename(tmp.name), data),
            model="whisper-large-v3-turbo",
        )
        out = (getattr(tr, "text", None) or "").strip()
        if len(out) < 5:
            return jsonify({"error": "Could not transcribe audio clearly."}), 422
        return jsonify({"text": out[:MAX_TEXT_LENGTH]})
    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e) or "Transcription failed"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/api/ocr-image", methods=["POST"])
def ocr_image():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    try:
        from PIL import Image
        import pytesseract

        tcmd = os.getenv("TESSERACT_CMD")
        if tcmd:
            pytesseract.pytesseract.tesseract_cmd = tcmd

        raw = request.files["image"].read()
        img = Image.open(io.BytesIO(raw))
        text = pytesseract.image_to_string(img)
        text = " ".join(text.split())
        if len(text) < MIN_TEXT_LENGTH:
            return jsonify({
                "error": "Not enough text detected. Try a sharper screenshot or paste text instead."
            }), 422
        return jsonify({"text": text[:MAX_TEXT_LENGTH]})
    except Exception as e:
        logger.exception("OCR failed")
        if e.__class__.__name__ == "TesseractNotFoundError":
            return jsonify({
                "error": "Tesseract OCR is not installed or not on PATH. Install from "
                "https://github.com/UB-Mannheim/tesseract/wiki or set TESSERACT_CMD in .env to tesseract.exe.",
            }), 503
        if "tesseract" in str(e).lower() and ("not found" in str(e).lower() or "path" in str(e).lower()):
            return jsonify({
                "error": "Tesseract executable not found. Add it to PATH or set TESSERACT_CMD in .env.",
            }), 503
        return jsonify({
            "error": "Image text extraction failed. Install Tesseract OCR or set TESSERACT_CMD."
        }), 500


# ── Auth ──────────────────────────────────────────────────────
@app.route("/api/me", methods=["GET"])
def me():
    ut = session.get("user_type")
    if ut == "guest":
        return jsonify({"user_type": "guest"})
    uid = session.get("user_id")
    if ut == "member" and uid:
        with dbstore.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, email, phone, display_name, created_at FROM users WHERE id = %s",
                    (int(uid),),
                )
                row = cur.fetchone()
                cur.execute(
                    "SELECT genre FROM user_genres WHERE user_id = %s ORDER BY genre",
                    (int(uid),),
                )
                genres = [r["genre"] for r in cur.fetchall()]
        if row:
            return jsonify({
                "user_type": "member",
                "email": row["email"],
                "phone": row["phone"],
                "display_name": row.get("display_name"),
                "genres": genres,
            })
    return jsonify({"user_type": None})


@app.route("/api/user/profile", methods=["GET", "PATCH"])
def user_profile():
    uid = session.get("user_id")
    if session.get("user_type") != "member" or not uid:
        return jsonify({"error": "Sign in required"}), 403
    uid = int(uid)
    if request.method == "GET":
        with dbstore.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT email, phone, display_name, created_at FROM users WHERE id = %s",
                    (uid,),
                )
                row = cur.fetchone()
        if not row:
            return jsonify({"error": "User not found"}), 404
        return jsonify(dict(row))
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("display_name") or "").strip()[:200]
    with dbstore.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET display_name = %s WHERE id = %s",
                (name or None, uid),
            )
    return jsonify({"ok": True, "display_name": name})


@app.route("/api/user/history", methods=["GET"])
def user_history():
    uid = session.get("user_id")
    if session.get("user_type") != "member" or not uid:
        return jsonify({"error": "Sign in required"}), 403
    limit = min(int(request.args.get("limit", 50)), 100)
    items = dbstore.fetch_history(int(uid), limit)
    # JSON-serialize datetimes
    for it in items:
        if it.get("created_at"):
            it["created_at"] = it["created_at"].isoformat()
    return jsonify({"history": items})


@app.route("/api/auth/guest", methods=["POST"])
def auth_guest():
    session.clear()
    session["user_type"] = "guest"
    session["user_id"] = None
    session["last_seen"] = datetime.now(timezone.utc).isoformat()
    session.permanent = True
    return jsonify({"ok": True, "user_type": "guest"})


@app.route("/api/auth/request-otp", methods=["POST"])
def auth_request_otp():
    data = request.get_json(force=True, silent=True) or {}
    channel = (data.get("channel") or "").lower()
    if channel == "email":
        email = (data.get("email") or "").strip().lower()
        if not email or "@" not in email:
            return jsonify({"error": "Valid email required"}), 400
        code = f"{secrets.randbelow(900000) + 100000}"
        dbstore.upsert_otp("email", email, code, None)
        sent = _send_email_otp(email, code)
        show_otp = os.getenv("AUTH_DEBUG_OTP", "").lower() in ("1", "true", "yes")
        msg = "OTP sent to your inbox." if sent else "Could not send email right now. Please try again."
        body = {"ok": True, "message": msg}
        if show_otp or not sent:
            body["dev_otp"] = code
        if not sent:
            logger.warning("OTP for %s (copy for testing): %s", email, code)
        return jsonify(body)

    return jsonify({"error": "channel must be email"}), 400


@app.route("/api/auth/verify-otp", methods=["POST"])
def auth_verify_otp():
    data = request.get_json(force=True, silent=True) or {}
    channel = (data.get("channel") or "").lower()
    otp = (data.get("otp") or "").strip()
    if not otp:
        return jsonify({"error": "OTP required"}), 400

    if channel == "email":
        email = (data.get("email") or "").strip().lower()
        rec = dbstore.verify_otp_row("email", email, otp)
        if not rec:
            return jsonify({"error": "Invalid or expired OTP — request a new one"}), 400
        with dbstore.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                existing = cur.fetchone()
                if existing:
                    uid = int(existing["id"])
                else:
                    cur.execute(
                        "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
                        (email, ""),
                    )
                    uid = int(cur.fetchone()["id"])
        session.clear()
        session["user_type"] = "member"
        session["user_id"] = uid
        session["email"] = email
        session["last_seen"] = datetime.now(timezone.utc).isoformat()
        session.permanent = True
        return jsonify({"ok": True, "user_type": "member"})
    return jsonify({"error": "Invalid channel"}), 400


@app.route("/api/auth/google", methods=["POST"])
def auth_google():
    data = request.get_json(force=True, silent=True) or {}
    token = data.get("credential") or data.get("token")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not token or not client_id:
        return jsonify({"error": "Google sign-in not configured (set GOOGLE_CLIENT_ID)"}), 503
    try:
        from google.oauth2 import id_token
        from google.auth.transport import requests as google_requests

        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), client_id
        )
        email = (idinfo.get("email") or "").lower()
        sub = idinfo.get("sub")
        if not email:
            return jsonify({"error": "No email on Google account"}), 400
        with dbstore.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id FROM users WHERE google_sub = %s OR email = %s",
                    (sub, email),
                )
                row = cur.fetchone()
                if row:
                    uid = int(row["id"])
                    cur.execute(
                        "UPDATE users SET google_sub = %s, email = %s WHERE id = %s",
                        (sub, email, uid),
                    )
                else:
                    cur.execute(
                        "INSERT INTO users (email, google_sub, password_hash) VALUES (%s, %s, %s) RETURNING id",
                        (email, sub, ""),
                    )
                    uid = int(cur.fetchone()["id"])
        session.clear()
        session["user_type"] = "member"
        session["user_id"] = uid
        session["email"] = email
        session["last_seen"] = datetime.now(timezone.utc).isoformat()
        session.permanent = True
        return jsonify({"ok": True, "user_type": "member"})
    except Exception as e:
        logger.warning("Google auth failed: %s", e)
        return jsonify({"error": "Google sign-in failed"}), 401


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/headlines", methods=["GET"])
def headlines():
    if session.get("user_type") != "member" or not session.get("user_id"):
        return jsonify({"error": "Available for signed-in members"}), 403

    country = _resolve_country_from_request(default_country="us")

    # Global major events (world / general) + localized top stories.
    local_articles = _newsapi_top(category="general", page_size=12, country=country)
    global_articles = _newsapi_top(q="world", page_size=10)

    # Local first, then global.
    merged = _merge_unique_by_url(local_articles, global_articles)
    if not merged:
        merged = _newsapi_top(category="general", page_size=14, country=country)
    return jsonify({"articles": merged[:20]})


ALLOWED_GENRES = (
    "technology",
    "business",
    "sports",
    "entertainment",
    "health",
    "science",
    "general",
)


@app.route("/api/user/genres", methods=["GET", "POST"])
def user_genres():
    uid = session.get("user_id")
    if session.get("user_type") != "member" or not uid:
        return jsonify({"error": "Sign in required"}), 403
    uid = int(uid)
    if request.method == "GET":
        with dbstore.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT genre FROM user_genres WHERE user_id = %s ORDER BY genre",
                    (uid,),
                )
                rows = cur.fetchall()
        return jsonify({"genres": [r["genre"] for r in rows]})

    data = request.get_json(force=True, silent=True) or {}
    genres = data.get("genres") or []
    if not isinstance(genres, list):
        return jsonify({"error": "genres must be a list"}), 400
    clean = [g for g in genres if g in ALLOWED_GENRES]
    with dbstore.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_genres WHERE user_id = %s", (uid,))
            for g in clean:
                cur.execute(
                    """INSERT INTO user_genres (user_id, genre) VALUES (%s, %s)
                       ON CONFLICT (user_id, genre) DO NOTHING""",
                    (uid, g),
                )
    return jsonify({"ok": True, "genres": clean})


@app.route("/api/news-personalized", methods=["GET"])
def news_personalized():
    uid = session.get("user_id")
    if session.get("user_type") != "member" or not uid:
        return jsonify({"error": "Sign in required"}), 403
    uid = int(uid)

    country = _resolve_country_from_request(default_country="us")

    with dbstore.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT genre FROM user_genres WHERE user_id = %s",
                (uid,),
            )
            rows = cur.fetchall()
    genres = [r["genre"] for r in rows] or ["general"]
    selected = genres[:5]
    seen = set()
    merged = []

    # Parallelize external API calls for much faster tab switch.
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(selected)))) as ex:
        fut_to_genre = {
            ex.submit(_genre_news_bundle, g, country): g
            for g in selected
        }
        for fut in as_completed(fut_to_genre):
            try:
                items = fut.result() or []
            except Exception:
                items = []
            for a in items:
                u = a.get("url") or ""
                if u and u not in seen:
                    seen.add(u)
                    merged.append(a)

    # Keep this genre-focused; avoid mixing generic world feed here.
    return jsonify({"articles": merged[:20]})


@app.route("/api/news-by-genre", methods=["GET"])
def news_by_genre():
    """
    Genre-first localized headlines (used by genres.html for previews).
    """
    uid = session.get("user_id")
    if session.get("user_type") != "member" or not uid:
        return jsonify({"error": "Sign in required"}), 403
    genre = (request.args.get("genre") or "").strip().lower()
    if genre not in ALLOWED_GENRES:
        return jsonify({"error": "Invalid genre"}), 400

    country = _resolve_country_from_request(default_country="us")

    # Location prioritized, with robust fallbacks for category sparsity.
    arts = _genre_news_bundle(genre, country)
    return jsonify({"articles": arts[:12]})


# ── Static frontend ───────────────────────────────────────────
@app.route("/")
def serve_landing():
    try:
        return send_from_directory(FRONTEND_DIR, "landing.html")
    except Exception as e:
        return f"Backend running 🚀 (Error: {str(e)})"


@app.route("/<path:fname>")
def serve_frontend(fname):
    try:
        if fname.startswith("api/"):
            abort(404)

        safe = os.path.normpath(fname).lstrip(".\\/")
        full = os.path.join(FRONTEND_DIR, safe)

        if not full.startswith(FRONTEND_DIR):
            abort(404)

        if os.path.isfile(full):
            return send_from_directory(FRONTEND_DIR, safe)

        # ✅ fallback → send landing page instead of crash
        return send_from_directory(FRONTEND_DIR, "landing.html")

    except Exception as e:
        return f"Error loading page: {str(e)}", 500


if __name__ == "__main__":
    logger.info("Starting VeritAI backend — open http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
