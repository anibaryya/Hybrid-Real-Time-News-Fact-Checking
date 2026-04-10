import sys
import os
import re
import io
import logging
import secrets
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_ANALYZE_NEWS_IMPORT_ERROR = None
try:
    from ml_model.model import analyze_news as _analyze_news_impl
except Exception as exc:
    _ANALYZE_NEWS_IMPORT_ERROR = exc
    logger.exception("Could not import ml_model.model.analyze_news; using degraded fallback")

    def analyze_news(text):
        return {
            "label": "UNCERTAIN",
            "confidence": 52.0,
            "certainty": "LOW",
            "reason": "Analysis engine is temporarily unavailable on the server.",
            "articles": [],
            "scores": {"bert": None, "tfidf": None, "evidence": 0.0, "judge": None},
            "followup_question": None,
            "followup_yes_prompt": None,
        }
else:
    analyze_news = _analyze_news_impl

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

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

from flask import Flask, request, jsonify, send_from_directory, session, abort

app = Flask(__name__)
@app.route("/")
def landing():
    return send_from_directory(FRONTEND_DIR, "landing.html")

@app.route("/index.html")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# Use a stable fallback so sessions survive backend restarts in local/dev runs.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "veritai-local-dev-secret-change-me"
app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_HTTPONLY=True,
    # Keep signed-in members logged in across browser restarts until they sign out.
    PERMANENT_SESSION_LIFETIME=timedelta(days=3650),
)
CORS(
    app,
    supports_credentials=True,
    # Railway/production-friendly: allow origins from env or any origin by default.
    # Example env: CORS_ORIGINS=https://your-frontend.up.railway.app,https://your-custom-domain.com
    resources={r"/api/*": {"origins": [o.strip() for o in (os.getenv("CORS_ORIGINS", "*")).split(",") if o.strip()]}},
)

MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 20

GUEST_TTL = timedelta(minutes=30)


def _normalize_analysis_result(result) -> dict:
    fallback = {
        "label": "UNCERTAIN",
        "confidence": 52.0,
        "certainty": "LOW",
        "reason": "Analysis engine is temporarily unavailable. Please try again shortly.",
        "articles": [],
        "scores": {"bert": None, "tfidf": None, "evidence": 0.0, "judge": None},
        "followup_question": None,
        "followup_yes_prompt": None,
    }

    if isinstance(result, dict):
        label = str(result.get("label") or fallback["label"]).upper()
        if label not in {"REAL", "FAKE", "UNCERTAIN"}:
            label = fallback["label"]

        certainty = str(result.get("certainty") or fallback["certainty"]).upper()
        if certainty not in {"LOW", "MEDIUM", "HIGH", "VERY LOW"}:
            certainty = fallback["certainty"]

        try:
            confidence = float(result.get("confidence", fallback["confidence"]))
        except (TypeError, ValueError):
            confidence = fallback["confidence"]

        articles = result.get("articles", fallback["articles"])
        if not isinstance(articles, list):
            articles = fallback["articles"]

        scores = result.get("scores", fallback["scores"])
        if not isinstance(scores, dict):
            scores = fallback["scores"]

        normalized = dict(fallback)
        normalized.update(
            {
                "label": label,
                "confidence": confidence,
                "certainty": certainty,
                "reason": str(result.get("reason") or fallback["reason"]),
                "articles": articles,
                "scores": scores,
                "followup_question": result.get("followup_question"),
                "followup_yes_prompt": result.get("followup_yes_prompt"),
            }
        )
        return normalized

    logger.warning("Analyzer returned unexpected result type %s: %r", type(result).__name__, result)
    if isinstance(result, str) and result.strip():
        fallback["reason"] = result.strip()
    elif result is not None:
        fallback["reason"] = f"Unexpected analyzer response type: {type(result).__name__}."

    if _ANALYZE_NEWS_IMPORT_ERROR is not None:
        fallback["reason"] = "Analysis engine is unavailable due to server configuration. Check Railway environment variables and redeploy."
    return fallback


@app.before_request
def _enforce_session_ttl():
    """
    - Guest sessions: expire after 30 minutes of inactivity.
    - Member sessions (email OTP / Google): stay active until manual logout.
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
    if ut == "guest" and last_seen and (now - last_seen) > GUEST_TTL:
        session.clear()
        return
    session["last_seen"] = now.isoformat()
    # Keep cookie across browser restarts; guests are still server-expired via GUEST_TTL.
    session.permanent = True


try:
    dbstore.init_db()
except Exception:
    logger.exception("Database init failed — set DATABASE_URL (PostgreSQL)")
    # Keep the service running in guest/demo mode even when PostgreSQL isn't available.
    # Routes that require DB will still fail if called, but /api/analyze-news can work.

# ── URL / article text ────────────────────────────────────────
ARTICLE_NOISE_RE = re.compile(
    r"(comment|related|recommend|trending|popular|newsletter|subscribe|promo|advert|"
    r"cookie|consent|social|share|footer|header|nav|menu|sidebar|aside|rail|widget|"
    r"taboola|outbrain|banner|popup|modal|read-more|most-read|breaking-news|tags|breadcrumb)",
    re.IGNORECASE,
)
ARTICLE_CONTENT_RE = re.compile(
    r"(article|story|content|post|entry|main|body|text|copy|article-body|story-body|article-content)",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _node_hints(node) -> str:
    if not node or not getattr(node, "attrs", None):
        return ""
    cls = " ".join(node.get("class", []) or [])
    return f"{node.get('id', '')} {cls}".strip()


def _extract_page_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return _normalize_text(og.get("content"))
    h1 = soup.find("h1")
    if h1:
        title = _normalize_text(h1.get_text(" ", strip=True))
        if title:
            return title
    if soup.title and soup.title.string:
        return _normalize_text(soup.title.string)
    return ""


def _extract_meta_description(soup: BeautifulSoup) -> str:
    desc_og = soup.find("meta", property="og:description")
    if desc_og and desc_og.get("content"):
        return _normalize_text(desc_og.get("content"))
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        return _normalize_text(md.get("content"))
    return ""


def _decompose_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form", "button", "nav", "footer", "aside", "header"]):
        tag.decompose()

    for node in list(soup.find_all(True)):
        hints = _node_hints(node)
        if not hints:
            continue
        if ARTICLE_NOISE_RE.search(hints) and not ARTICLE_CONTENT_RE.search(hints):
            node.decompose()


def _has_noisy_ancestor(node, stop_node) -> bool:
    parent = getattr(node, "parent", None)
    while parent and parent is not stop_node:
        hints = _node_hints(parent)
        if hints and ARTICLE_NOISE_RE.search(hints) and not ARTICLE_CONTENT_RE.search(hints):
            return True
        parent = getattr(parent, "parent", None)
    return False


def _dedupe_chunks(chunks: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = _normalize_text(chunk)
        if len(text) < 45:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _extract_paragraphs(node, min_len: int = 60) -> list[str]:
    chunks: list[str] = []
    for p in node.find_all("p"):
        if _has_noisy_ancestor(p, node):
            continue
        text = _normalize_text(p.get_text(" ", strip=True))
        if len(text) < min_len:
            continue
        lowered = text.lower()
        if any(phrase in lowered for phrase in ("subscribe", "sign up", "follow us", "read more", "advertisement")):
            continue
        chunks.append(text)
    return _dedupe_chunks(chunks)


def _score_candidate(node) -> tuple[int, list[str]]:
    paragraphs = _extract_paragraphs(node)
    if not paragraphs:
        return 0, []

    hints = _node_hints(node)
    score = sum(min(len(p), 420) for p in paragraphs[:16])
    score += min(len(paragraphs), 10) * 120
    if getattr(node, "name", "") == "article":
        score += 800
    elif getattr(node, "name", "") == "main":
        score += 500
    elif node.get("role") == "main":
        score += 350
    if ARTICLE_CONTENT_RE.search(hints):
        score += 220
    if ARTICLE_NOISE_RE.search(hints) and not ARTICLE_CONTENT_RE.search(hints):
        score -= 600
    return score, paragraphs


def _extract_article_payload_from_html(html: str, url: str, max_chars: int = MAX_TEXT_LENGTH) -> dict | None:
    soup = BeautifulSoup(html or "", "html.parser")
    title = _extract_page_title(soup)
    meta_desc = _extract_meta_description(soup)
    _decompose_noise(soup)

    candidates = []
    seen_nodes: set[int] = set()
    for node in soup.find_all(True):
        hints = _node_hints(node)
        if (
            node.name in {"article", "main"}
            or node.get("role") == "main"
            or node.get("itemprop") == "articleBody"
            or ARTICLE_CONTENT_RE.search(hints)
        ):
            ident = id(node)
            if ident in seen_nodes:
                continue
            seen_nodes.add(ident)
            candidates.append(node)

    if soup.body:
        candidates.append(soup.body)

    best_paragraphs: list[str] = []
    best_score = -1
    for node in candidates:
        score, paragraphs = _score_candidate(node)
        if score > best_score:
            best_score = score
            best_paragraphs = paragraphs

    if not best_paragraphs:
        best_paragraphs = _extract_paragraphs(soup, min_len=45)

    body = " ".join(best_paragraphs)
    if title and title.lower() not in body.lower():
        body = f"{title}. {body}".strip(". ")
    if len(body) < 220 and meta_desc and meta_desc.lower() not in body.lower():
        body = f"{body} {meta_desc}".strip()
    body = _normalize_text(body)[:max_chars]

    if not body:
        if not meta_desc:
            return None
        body = meta_desc[:max_chars]

    return {
        "title": title or "Article",
        "excerpt": body,
        "url": url,
    }


def _fetch_article_payload(url: str, timeout: int = 12, max_chars: int = MAX_TEXT_LENGTH) -> dict | None:
    res = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
        timeout=timeout,
    )
    res.raise_for_status()
    return _extract_article_payload_from_html(res.text, url, max_chars=max_chars)


def extract_text_from_url(url: str) -> str | None:
    try:
        payload = _fetch_article_payload(url, timeout=10, max_chars=MAX_TEXT_LENGTH)
        return payload["excerpt"] if payload and payload.get("excerpt") else None
    except requests.exceptions.RequestException as e:
        logger.error("URL fetch error for %s: %s", url, e)
        return None
    except Exception as e:
        logger.error("URL extraction error for %s: %s", url, e)
        return None


def extract_article_preview(url: str, max_chars: int = 4000) -> dict | None:
    """Title + lead paragraphs for in-app 'read more'."""
    try:
        payload = _fetch_article_payload(url, timeout=12, max_chars=max_chars)
        if not payload:
            return None
        return payload
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


def _gnews_top(
    topic: str | None = None,
    q: str | None = None,
    page_size: int = 12,
    country: str = "us",
) -> list:
    key = os.getenv("GNEWS_API_KEY")
    if not key:
        return []
    try:
        if topic:
            url = (
                "https://gnews.io/api/v4/top-headlines"
                f"?topic={requests.utils.quote(topic)}"
                f"&country={requests.utils.quote(country)}"
                f"&lang=en&max={page_size}&token={key}"
            )
        else:
            qq = q or "world"
            url = (
                "https://gnews.io/api/v4/search"
                f"?q={requests.utils.quote(qq)}"
                f"&country={requests.utils.quote(country)}"
                f"&lang=en&max={page_size}&token={key}"
            )
        r = requests.get(url, timeout=8).json()
        out = []
        for a in r.get("articles", []) or []:
            if not a.get("title"):
                continue
            source = a.get("source") or {}
            out.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": source.get("name", "") if isinstance(source, dict) else str(source or ""),
                "description": (a.get("description") or "").strip(),
                "image": a.get("image") or "",
                "publishedAt": a.get("publishedAt") or "",
            })
        return out[:page_size]
    except Exception as e:
        logger.warning("GNews headlines failed: %s", e)
        return []


def _google_news_rss(query: str | None = None, country: str = "us", page_size: int = 12) -> list:
    country = (country or "us").strip().lower()
    upper = country.upper() if len(country) == 2 else "US"
    hl = {
        "in": "en-IN",
        "gb": "en-GB",
        "au": "en-AU",
        "ca": "en-CA",
    }.get(country, f"en-{upper}")
    ceid = f"{upper}:en"

    if query:
        url = (
            "https://news.google.com/rss/search"
            f"?q={requests.utils.quote(query)}&hl={hl}&gl={upper}&ceid={ceid}"
        )
    else:
        url = f"https://news.google.com/rss?hl={hl}&gl={upper}&ceid={ceid}"

    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            source_el = item.find("source")
            source = (source_el.text or "").strip() if source_el is not None and source_el.text else "Google News"
            desc_html = item.findtext("description") or ""
            description = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)
            out.append({
                "title": title,
                "url": link,
                "source": source,
                "description": description,
                "image": "",
                "publishedAt": (item.findtext("pubDate") or "").strip(),
            })
        return out[:page_size]
    except Exception as e:
        logger.warning("Google News RSS failed: %s", e)
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
    a4 = _gnews_top(topic=cat, page_size=8, country=country)
    if a4:
        return a4
    a5 = _gnews_top(q=f"{genre} news", page_size=8, country=country)
    if a5:
        return a5
    a6 = _google_news_rss(query=f"{genre} news", country=country, page_size=8)
    if a6:
        return a6
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
    return jsonify(
        {
            "status": "ok",
            "keys": keys,
            "analyzer_import_ok": _ANALYZE_NEWS_IMPORT_ERROR is None,
        }
    ), 200


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
        context_notes = (request.form.get("context_notes") or "").strip()

        if not text and url:
            text = extract_text_from_url(url)
            if not text:
                return jsonify({"error": "Could not extract text from the provided URL."}), 422

        if context_notes:
            text = f"{text}\n\nAdditional timeline details: {context_notes}".strip()

        if not text:
            return jsonify({"error": "Please provide news text or a valid URL."}), 400

        if len(text) < MIN_TEXT_LENGTH:
            return jsonify({"error": f"Input too short (minimum {MIN_TEXT_LENGTH} characters)."}), 400

        text = text[:MAX_TEXT_LENGTH]
        input_mode = (request.form.get("input_mode") or "text").strip()[:24] or "text"
        try:
            result = analyze_news(text)
        except Exception:
            # Railway/production resilience: return a safe degraded verdict
            # instead of hard 500 so frontend doesn't drop to demo mode.
            logger.exception("analyze_news failed; returning degraded verdict")
            result = {
                "label": "UNCERTAIN",
                "confidence": 52.0,
                "certainty": "LOW",
                "reason": "Live verification is temporarily limited. Please retry in a moment.",
                "articles": [],
                "scores": {"bert": None, "tfidf": None, "evidence": 0.0, "judge": None},
                "followup_question": None,
                "followup_yes_prompt": None,
            }
        result = _normalize_analysis_result(result)

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
    country = _resolve_country_from_request(default_country="us")

    # Global major events (world / general) + localized top stories.
    local_articles = (
        _newsapi_top(category="general", page_size=12, country=country)
        or _gnews_top(topic="general", page_size=12, country=country)
        or _google_news_rss(country=country, page_size=12)
    )
    global_articles = (
        _newsapi_top(q="world", page_size=10)
        or _gnews_top(q="world", page_size=10, country=country)
        or _google_news_rss(query="world", country=country, page_size=10)
    )

    # Local first, then global.
    merged = _merge_unique_by_url(local_articles, global_articles)
    if not merged:
        merged = (
            _newsapi_top(category="general", page_size=14, country=country)
            or _gnews_top(topic="general", page_size=14, country=country)
            or _google_news_rss(country=country, page_size=14)
        )
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
    if fname.startswith("api/"):
        abort(404)

    safe = os.path.normpath(fname).lstrip(".\\/")
    full = os.path.join(FRONTEND_DIR, safe)

    if not full.startswith(FRONTEND_DIR):
        abort(404)

    if os.path.isfile(full):
        return send_from_directory(FRONTEND_DIR, safe)

    # fallback to landing page
    return send_from_directory(FRONTEND_DIR, "landing.html")

if __name__ == "__main__":
    logger.info("Starting VeritAI backend — open http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
