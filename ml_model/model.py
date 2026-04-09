import os
import re
import pickle
import logging
import requests
from urllib.parse import urlparse
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from bs4 import BeautifulSoup

try:
    from groq import Groq
except Exception:
    Groq = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────
# API KEYS
# ─────────────────────────────
GROQ_API_KEY   = (os.getenv("GROQ_API_KEY") or "").strip()
NEWS_API_KEY   = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY  = os.getenv("GNEWS_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

groq_client = None
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set; LLM-assisted analysis features disabled.")
elif Groq is None:
    logger.warning("groq package unavailable; LLM-assisted analysis features disabled.")
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        logger.warning(f"Groq client init failed; LLM-assisted analysis features disabled: {e}")


BASE = os.path.dirname(__file__)

# ─────────────────────────────
# LOAD MODELS  (with error handling)
# ─────────────────────────────
tfidf_model  = None
bert_model   = None
tokenizer    = None

try:
    with open(os.path.join(BASE, "tfidf_model.pkl"), "rb") as f:
        tfidf_model = pickle.load(f)
    logger.info("TF-IDF model loaded.")
except Exception as e:
    logger.warning(f"TF-IDF model not found, skipping: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    _bert_path = os.path.join(BASE, "bert_model")
    tokenizer  = AutoTokenizer.from_pretrained(_bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(_bert_path)
    bert_model.eval()
    logger.info("BERT model loaded.")
except Exception as e:
    logger.warning(f"BERT model not found, skipping: {e}")

# ─────────────────────────────
# TRUSTED SOURCES
# ─────────────────────────────
TRUSTED_SOURCES = [
    "bbc", "reuters", "ndtv", "times of india",
    "the hindu", "indian express", "associated press", "ap news",
    "anandabazar", "bartaman", "jagran", "hindustan times",
]

# ─────────────────────────────
# WEIGHTS  (base — redistributed dynamically when models are unavailable)
# ─────────────────────────────
# Favor retrieval + judge over headline-only classifiers (reduces false FAKE on real stories)
W_BERT      = 0.22
W_TFIDF     = 0.13
W_EVIDENCE  = 0.38   # APIs + scrape consensus
W_JUDGE     = 0.27   # LLM over evidence

def dynamic_weights(bert_ok: bool, tfidf_ok: bool, judge_ok: bool) -> dict:
    """
    Redistribute weight away from unavailable components so a missing model
    doesn't silently drag the final score toward 0 (fake) or 1 (real).
    """
    weights = {
        "bert":     W_BERT     if bert_ok  else 0.0,
        "tfidf":    W_TFIDF    if tfidf_ok else 0.0,
        "evidence": W_EVIDENCE,
        "judge":    W_JUDGE    if judge_ok else 0.0,
    }
    total = sum(weights.values())
    if total == 0:
        # last resort: everything rides on evidence
        return {"bert": 0.0, "tfidf": 0.0, "evidence": 1.0, "judge": 0.0}
    return {k: v / total for k, v in weights.items()}

# ─────────────────────────────
# PREPROCESS
# ─────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────
# TRANSLATION
# ─────────────────────────────
def is_english(text: str) -> bool:
    # heuristic: if >80% chars are ASCII, treat as English
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(len(text), 1) > 0.80

def translate(text: str) -> str:
    if is_english(text):
        return text
    if groq_client is None:
        return text
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    "Translate the following text to English. "
                    "Return ONLY the translated text, no explanation.\n\n"
                    f"{text[:2000]}"
                )
            }],
            temperature=0.1,
            max_tokens=1000,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text


def detect_language_code(text: str) -> str:
    """
    Light-weight language detection for this app.
    Returns: 'en', 'hi' (Devanagari), 'bn' (Bengali), or 'en' fallback.
    """
    if not text:
        return "en"

    # Use character counts instead of "exists" checks, and prioritize Bengali.
    # This prevents mixed-script inputs from being classified as Hindi just because
    # a few Devanagari characters appear (e.g., punctuation/formatting artifacts).
    bn_chars = len(re.findall(r"[\u0980-\u09FF]", text))
    hi_chars = len(re.findall(r"[\u0900-\u097F]", text))

    if bn_chars > 0 or hi_chars > 0:
        if bn_chars >= hi_chars:
            return "bn"
        return "hi"
    return "en" if is_english(text) else "en"


def translate_to_language(text: str, lang_code: str) -> str:
    if not text or lang_code == "en":
        return text
    if groq_client is None:
        return text
    lang_name = {"hi": "Hindi", "bn": "Bengali"}.get(lang_code, "English")
    try:
        prompt = (
            f"Translate the following text to {lang_name}. "
            "Return ONLY the translated text, with no quotes and no extra commentary.\n\n"
            f"{text[:2500]}"
        )
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=700,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(f"Back-translation failed: {e}")
        return text


def _looks_ambiguous_for_fact_check(original_text: str, translated_text: str) -> bool:
    """
    Heuristic gate: only if this returns True do we ask the user to clarify.
    """
    if not original_text or not translated_text:
        return True
    t = original_text.strip()
    t_en = translated_text.strip()
    if len(t) < 80:
        # Short claims are often underspecified for a fact-check.
        return True
    # Generic / umbrella phrasing without concrete identifiers.
    generic_terms = ("breaking", "news", "update", "rumor", "viral", "shocking", "is it true", "fact", "verify")
    if any(term in t.lower() for term in generic_terms) and not any(c.isdigit() for c in t_en):
        return True
    # If translated text has very few tokens, it's likely not specific.
    if len(t_en.split()) < 12:
        return True
    return False


def _ask_clarification_question(original_text: str, lang_code: str) -> str | None:
    """
    Ask a single follow-up question in the SAME language as the input,
    so the user can provide a precise, fact-checkable claim.
    """
    if groq_client is None:
        return None
    try:
        lang_name = {"hi": "Hindi", "bn": "Bengali", "en": "English"}.get(lang_code, "English")
        prompt = (
            "You are a helpful assistant for a news fact-checking tool.\n"
            "Determine whether the USER_INPUT is specific enough to fact-check.\n"
            "If it is too vague/generic and could refer to multiple different events/outcomes, "
            "you MUST ask one concise clarifying question to get missing specifics.\n\n"
            "Return ONLY this format:\n"
            "NEED_CLARIFICATION: YES|NO\n"
            "QUESTION: <question in the SAME language as the user's input> (only if YES)\n\n"
            f"USER_INPUT_LANGUAGE: {lang_name}\n\n"
            f"USER_INPUT: {original_text[:1200]}"
        )
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=160,
        )
        raw = (res.choices[0].message.content or "").strip()
        need = None
        question = None
        for line in raw.splitlines():
            if line.upper().startswith("NEED_CLARIFICATION:"):
                need = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip()
        if need == "YES" and question:
            return question
    except Exception as e:
        logger.warning(f"Clarification question failed: {e}")
    return None


def _build_binary_followup_question(original_text: str, lang_code: str) -> str:
    """
    Build a claim-specific YES/NO-friendly follow-up question in the same language
    as the user's input. Falls back to a generic template if generation fails.
    """
    fallback = translate_to_language(
        "I verified this against current reports. "
        "Are you referring to the current ongoing incident, or a past incident/time period?",
        lang_code,
    )
    if groq_client is None:
        return fallback
    try:
        lang_name = {"hi": "Hindi", "bn": "Bengali", "en": "English"}.get(lang_code, "English")
        prompt = (
            "Rewrite the follow-up into a claim-specific question based on USER_INPUT.\n"
            "Requirements:\n"
            "1) Keep it YES/NO-friendly.\n"
            "2) Mention the core claim/entity from USER_INPUT (not generic wording).\n"
            "3) Ask whether the user refers to current incident vs past incident/time period.\n"
            "4) Output in the same language as USER_INPUT.\n"
            "5) Return ONLY one question line.\n\n"
            f"USER_INPUT_LANGUAGE: {lang_name}\n"
            f"USER_INPUT: {original_text[:1200]}"
        )
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        q = (res.choices[0].message.content or "").strip()
        if not q:
            return fallback
        # Guardrail: keep question concise and single-line.
        q = " ".join(q.split())
        if len(q) < 18:
            return fallback
        return q
    except Exception as e:
        logger.warning(f"Binary follow-up generation failed: {e}")
        return fallback

# ─────────────────────────────
# QUERY GENERATION
# ─────────────────────────────
def generate_query(text: str) -> str:
    if groq_client is None:
        return " ".join(text.split()[:10])
    try:
        prompt = (
            "Extract a SHORT factual search query (max 10 words) from the text below. "
            "Include the key event, location, and person if present. "
            "Return ONLY the query string.\n\n"
            f"Text: {text[:500]}"
        )
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=50,
        )
        query = res.choices[0].message.content.strip().strip('"')
        logger.info(f"Generated query: {query}")
        return query
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
        # fallback: first 10 words of text
        return " ".join(text.split()[:10])

# ─────────────────────────────
# ML PREDICTIONS
# ─────────────────────────────
def predict_tfidf(text: str) -> float | None:
    """Returns probability that text is REAL (0–1), or None if model unavailable."""
    if tfidf_model is None:
        return None
    try:
        probs = tfidf_model.predict_proba([preprocess(text)])[0]
        return float(probs[1])  # index 1 = REAL
    except Exception as e:
        logger.warning(f"TF-IDF prediction failed: {e}")
        return None

def predict_bert(text: str) -> float | None:
    """Returns probability that text is REAL (0–1), or None if model unavailable."""
    if bert_model is None or tokenizer is None:
        return None
    try:
        import torch
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, padding=True, max_length=128
        )
        with torch.no_grad():
            logits = bert_model(**inputs).logits
            probs  = torch.nn.functional.softmax(logits, dim=1)
        return float(probs[0][1])  # index 1 = REAL
    except Exception as e:
        logger.warning(f"BERT prediction failed: {e}")
        return None

# ─────────────────────────────
# NEWS APIs  (lru_cache needs hashable args — strings are fine)
# ─────────────────────────────
@lru_cache(maxsize=128)
def fetch_news(query: str) -> tuple:
    """Returns a tuple of article dicts (hashable for lru_cache)."""
    if not NEWS_API_KEY:
        return ()
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={requests.utils.quote(query)}"
            f"&apiKey={NEWS_API_KEY}&pageSize=5&sortBy=relevancy"
        )
        res = requests.get(url, timeout=5).json()
        articles = [
            {
                "title":       a.get("title", ""),
                "url":         a.get("url", ""),
                "source":      a.get("source", {}).get("name", ""),
                "description": (a.get("description") or "").strip(),
                "image":       a.get("urlToImage") or "",
                "publishedAt": a.get("publishedAt") or "",
            }
            for a in res.get("articles", [])[:5]
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
        return tuple(articles)   # ← tuple so lru_cache works
    except Exception as e:
        logger.warning(f"NewsAPI failed: {e}")
        return ()

@lru_cache(maxsize=128)
def fetch_gnews(query: str) -> tuple:
    """Returns (evidence_text, tuple of article dicts)."""
    if not GNEWS_API_KEY:
        return ("", ())
    try:
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={requests.utils.quote(query)}&token={GNEWS_API_KEY}&max=5"
        )
        res = requests.get(url, timeout=5).json()
        raw = res.get("articles", [])[:5]

        def _src_name(a):
            s = a.get("source")
            if isinstance(s, dict):
                return s.get("name", "") or ""
            if isinstance(s, str):
                return s
            return ""

        articles = tuple(
            {
                "title":       a.get("title", ""),
                "url":         a.get("url", ""),
                "source":      _src_name(a),
                "description": (a.get("description") or "").strip(),
                "image":       a.get("image") or "",
                "publishedAt": a.get("publishedAt") or "",
            }
            for a in raw
            if a.get("title")
        )
        text = " ".join(a.get("title", "") + " " + _src_name(a) for a in raw)
        return (text, articles)
    except Exception as e:
        logger.warning(f"GNews failed: {e}")
        return ("", ())

# ─────────────────────────────
# SERPER (REAL-TIME SEARCH)
# ─────────────────────────────
def search_serper(query: str) -> str:
    text, _ = search_serper_with_results(query)
    return text


def search_serper_with_results(query: str) -> tuple[str, list]:
    """Evidence text + organic result dicts (title, link, snippet)."""
    if not SERPER_API_KEY:
        return "", []
    try:
        res = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 8},
            timeout=6,
        ).json()
        organic = res.get("organic", []) or []
        snippets = [
            item.get("title", "") + " " + item.get("snippet", "") + " " + item.get("link", "")
            for item in organic[:8]
        ]
        return " ".join(snippets), organic[:8]
    except Exception as e:
        logger.warning(f"Serper failed: {e}")
        return "", []


def _merge_article_lists(*lists: list) -> list:
    seen: set[str] = set()
    out: list = []
    for lst in lists:
        for a in lst or []:
            u = (a.get("url") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(dict(a))
    return out[:12]

# ─────────────────────────────
# REGIONAL SCRAPING
# ─────────────────────────────
def scrape_regional_news(query: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://www.anandabazar.com/search?q={requests.utils.quote(query)}",
        f"https://www.jagran.com/search/{requests.utils.quote(query)}",
    ]
    texts = []
    for url in urls:
        try:
            res  = requests.get(url, headers=headers, timeout=4)
            soup = BeautifulSoup(res.text, "html.parser")
            for p in soup.find_all("p"):
                t = p.get_text(strip=True)
                if len(t) > 40:
                    texts.append(t)
                    if len(texts) >= 8:
                        break
        except Exception as e:
            logger.debug(f"Regional scrape failed for {url}: {e}")
    return " ".join(texts)

# ─────────────────────────────
# SOURCE CONSENSUS
# ─────────────────────────────
def source_consensus(evidence: str) -> int:
    ev_lower = evidence.lower()
    return sum(1 for s in TRUSTED_SOURCES if s in ev_lower)

def recency_boost(evidence: str) -> float:
    keywords = ["today", "breaking", "just in", "latest", "hours ago"]
    return 0.1 if any(k in evidence.lower() for k in keywords) else 0.0

# ─────────────────────────────
# EVIDENCE COLLECTION
# ─────────────────────────────
def get_evidence(query: str):
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures["news"]     = ex.submit(fetch_news, query)
        futures["gnews"]    = ex.submit(fetch_gnews, query)
        futures["serper"]   = ex.submit(search_serper_with_results, query)
        futures["regional"] = ex.submit(scrape_regional_news, query)

    articles_news = list(futures["news"].result())
    gnews_txt, gnews_articles = futures["gnews"].result()
    serper_txt, serper_organic = futures["serper"].result()
    regional  = futures["regional"].result()

    serper_as_articles = []
    for item in serper_organic or []:
        link = item.get("link") or ""
        if not link:
            continue
        try:
            host = urlparse(link).netloc.replace("www.", "")
        except Exception:
            host = "web"
        serper_as_articles.append({
            "title":       item.get("title") or "",
            "url":         link,
            "source":      host,
            "description": (item.get("snippet") or "").strip(),
            "image":       "",
            "publishedAt": "",
        })

    merged_articles = _merge_article_lists(
        articles_news,
        list(gnews_articles),
        serper_as_articles,
    )

    news_text = " ".join(
        a["title"] + " " + a.get("source", "") for a in articles_news
    )
    evidence = f"{news_text} {gnews_txt} {serper_txt} {regional}"

    # cap evidence to ~3000 chars to stay within Groq context
    evidence_for_judge = evidence[:3000]

    consensus = source_consensus(evidence)
    recency   = recency_boost(evidence)

    return evidence_for_judge, merged_articles, consensus, recency

# ─────────────────────────────
# AI JUDGE — returns (reason: str, score: float | None, verdict: str | None)
# judge_score: 1.0 = REAL, 0.0 = FAKE, 0.5 = UNCERTAIN
# ─────────────────────────────
def judge(claim: str, evidence: str, articles: list[dict]):
    """
    Returns (reason: str, score: float | None, verdict: str | None, evidence_indices: list[int] | None).
    score=1.0 REAL, 0.0 FAKE, 0.5 UNCERTAIN, None = call failed.
    """
    if groq_client is None:
        return "AI judge unavailable.", None, None, None
    try:
        # Used to bias verdicts toward FAKE for claims about events that already happened.
        now_year = datetime.now(timezone.utc).year
        past_keywords = ("yesterday", "today", "last year", "last month", "ago")
        claim_years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", claim)]
        is_past_claim = any(y <= now_year for y in claim_years) and any(k in claim.lower() for k in past_keywords) or any(
            y <= now_year - 1 for y in claim_years
        )

        # Provide the judge with a small, enumerated list of candidate sources
        # so it can return indices that actually support/contradict the claim.
        limited_articles = list(articles or [])[:8]
        articles_block = "\n".join(
            f"[{i}] {a.get('title','').strip()} | {a.get('source','').strip()} | {a.get('description','').strip()}"
            for i, a in enumerate(limited_articles)
        ) or "(no candidate sources)"

        prompt = (
            "You are a news fact-checking AI.\n"
            "Given the CLAIM and EVIDENCE (news API + web snippets), decide:\n"
            "- REAL : if credible sources describe the same event/claim or clearly support it "
            "(even with different wording).\n"
            "- FAKE : if evidence clearly contradicts key details (people, places, numbers, outcomes, or dates) "
            "OR if evidence explicitly debunks the claim.\n"
            "- UNCERTAIN : only when evidence is missing, off-topic, or does not contain enough information to decide.\n"
            "Never output UNCERTAIN when the evidence contains a clear contradiction.\n\n"
            f"CLAIM_APPEARS_TO_BE_PAST_EVENT: {'YES' if is_past_claim else 'NO'}\n"
            "If CLAIM_APPEARS_TO_BE_PAST_EVENT is YES and evidence indicates it did not happen "
            "or contradicts the stated details, output FAKE.\n\n"
            "Respond in this exact format:\n"
            "VERDICT: <REAL|FAKE|UNCERTAIN>\n"
            "REASON: <one or two sentences>\n\n"
            "EVIDENCE_INDICES: <comma-separated indices from the provided candidate sources, max 5>\n\n"
            f"CLAIM: {claim[:800]}\n\n"
            f"EVIDENCE: {evidence[:2000]}\n\n"
            f"Candidate sources (use indices):\n{articles_block}"
        )
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = res.choices[0].message.content.strip()

        verdict_line = None
        reason_line = ""
        for line in raw.splitlines():
            if line.upper().startswith("VERDICT:"):
                verdict_line = line.split(":", 1)[1].strip().upper()
            elif line.upper().startswith("REASON:"):
                reason_line = line.split(":", 1)[1].strip()

        if not verdict_line:
            m = re.search(r"VERDICT:\s*(REAL|FAKE|UNCERTAIN)", raw, flags=re.IGNORECASE)
            verdict_line = m.group(1).upper() if m else "UNCERTAIN"

        score = 0.5  # UNCERTAIN
        if verdict_line == "REAL":
            score = 1.0
        elif verdict_line == "FAKE":
            score = 0.0
        evidence_indices = None
        try:
            # More robust than line-prefix matching (handles spaces, brackets, etc.)
            m_idx = re.search(r"EVIDENCE_INDICES\s*:\s*([^\n]+)", raw, flags=re.IGNORECASE)
            indices_str = (m_idx.group(1) if m_idx else "").strip()
            if indices_str:
                ints = [int(d) for d in re.findall(r"\d+", indices_str)]
                # Keep only indices that exist in the limited list.
                if ints:
                    evidence_indices = sorted(set(i for i in ints if 0 <= i < len(limited_articles)))[:5]
        except Exception:
            evidence_indices = None

        return reason_line or raw, score, verdict_line, evidence_indices

    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return "AI judge unavailable.", None, None, None   # None signals: exclude from scoring


def _claim_looks_past(text: str) -> bool:
    """Heuristic: does the claim reference an event that already happened?"""
    now_year = datetime.now(timezone.utc).year
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    if any(y <= now_year - 1 for y in years):
        return True
    # light keyword heuristic
    t = text.lower()
    return any(k in t for k in ("yesterday", "last year", "last month", "ago"))

# ─────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────
def analyze_news(text: str) -> dict:
    original_text = text
    input_lang = detect_language_code(original_text)

    # 1. Translate if needed
    text = translate(text)

    # 2b. Ambiguous-claim follow-up:
    # We still produce a verdict against currently occurring/reported incidents,
    # then ask whether the user meant a different incident/time period.
    followup_question = None
    followup_yes_prompt = None
    if _looks_ambiguous_for_fact_check(original_text, text):
        # Keep the follow-up strictly YES/NO-compatible and claim-specific.
        followup_question = _build_binary_followup_question(original_text, input_lang)
        followup_yes_prompt = translate_to_language(
            "Please share the timeline and detailed incident info "
            "(date/time, location, people involved, and a source link if possible).",
            input_lang,
        )

    # 2. ML scores  (None = model not available)
    tfidf_score = predict_tfidf(text)
    bert_score  = predict_bert(text)

    # 3. Evidence retrieval
    query = generate_query(text)
    evidence, articles, consensus, recency = get_evidence(query)
    n_articles = len(articles)
    is_past_claim = _claim_looks_past(text)

    # 4. Evidence score — boost when APIs/scraping returned multiple distinct articles
    base_ev = (consensus / max(len(TRUSTED_SOURCES), 1)) + recency
    article_boost = min(n_articles * 0.065, 0.36)
    evidence_score = min(base_ev + article_boost, 1.0)

    # 5. AI Judge  (None score = call failed → excluded from weighting)
    judge_reason, judge_score, _judge_verdict, evidence_indices = judge(text, evidence, articles)

    judge_reason_out = translate_to_language(judge_reason, input_lang)

    # 6. Dynamic weights — skip components whose model is unavailable
    w = dynamic_weights(
        bert_ok  = bert_score  is not None,
        tfidf_ok = tfidf_score is not None,
        judge_ok = judge_score is not None,
    )

    final_score = (
        w["bert"]     * (bert_score     if bert_score  is not None else 0) +
        w["tfidf"]    * (tfidf_score    if tfidf_score is not None else 0) +
        w["evidence"] * evidence_score +
        w["judge"]    * (judge_score    if judge_score is not None else 0)
    )

    # 7. Label + certainty (raw from blended score)
    if final_score >= 0.65:
        label = "REAL"
        certainty = "HIGH" if final_score >= 0.80 else "MEDIUM"
    elif final_score <= 0.35:
        label = "FAKE"
        certainty = "HIGH" if final_score <= 0.20 else "MEDIUM"
    else:
        label = "UNCERTAIN"
        certainty = "LOW"

    # 8. Policy: if the judge detects a clear contradiction, force FAKE.
    # This fixes the "contradicting news but returns UNCERTAIN" failure mode.
    corroborated_web = n_articles >= 2 or evidence_score >= 0.20
    if judge_score == 1.0:
        label = "REAL"
        if final_score < 0.58:
            certainty = "LOW"
        elif final_score < 0.76:
            certainty = "MEDIUM"
        else:
            certainty = "HIGH" if final_score >= 0.85 else "MEDIUM"
    elif judge_score == 0.0:
        label = "FAKE"
        if is_past_claim and corroborated_web:
            certainty = "HIGH"
        elif corroborated_web:
            certainty = "MEDIUM"
        else:
            # Judge found contradictions but evidence is weak; keep certainty conservative.
            certainty = "LOW"
    elif judge_score == 0.5 and corroborated_web:
        label = "REAL"
        certainty = "LOW"
    elif judge_score == 0.5 and n_articles >= 1 and evidence_score >= 0.12:
        label = "REAL"
        certainty = "LOW"

    # Confidence should reflect the *predicted* class.
    # If we predict FAKE, use (1-final_score) directionality.
    if label == "REAL":
        label_conf = final_score
    else:
        label_conf = 1.0 - final_score

    # Ensure contradictions get at least a modest confidence floor.
    if label == "FAKE" and judge_score == 0.0:
        if certainty == "HIGH":
            label_conf = max(label_conf, 0.75)
        elif certainty == "MEDIUM":
            label_conf = max(label_conf, 0.65)
        else:
            label_conf = max(label_conf, 0.55)

    confidence = round(min(label_conf * 100, 99.9), 1)

    # Keep only the judge-selected sources (or fall back to full list if parsing failed).
    selected_articles = articles
    if evidence_indices is not None:
        try:
            selected_articles = [articles[i] for i in evidence_indices if 0 <= i < len(articles)]
            if not selected_articles:
                selected_articles = articles
        except Exception:
            selected_articles = articles

    logger.info(
        f"Result → label={label} confidence={confidence}% "
        f"[bert={bert_score} tfidf={tfidf_score} "
        f"evidence={evidence_score:.2f} judge={judge_score} n_art={n_articles}] "
        f"weights={w}"
    )

    return {
        "label":      label,
        "confidence": confidence,
        "certainty":  certainty,
        "reason":     judge_reason_out,
        "followup_question": followup_question,
        "followup_yes_prompt": followup_yes_prompt,
        # Show evidence snippets (judge-selected) for both REAL and FAKE outputs.
        "articles":   selected_articles,
        "scores": {
            "bert":     round(bert_score,  3) if bert_score  is not None else None,
            "tfidf":    round(tfidf_score, 3) if tfidf_score is not None else None,
            "evidence": round(evidence_score, 3),
            "judge":    round(judge_score, 3) if judge_score is not None else None,
        }
    }
