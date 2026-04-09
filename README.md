# 🔍 VeritAI — Real-Time AI News Fact Checker & Judge

> **B.Tech Final Year Project | Computer Science | 2024–25**

---

## 📌 Project Overview

**VeritAI** is an advanced AI-powered system that verifies the authenticity of news using a hybrid approach combining:

* 🧠 Machine Learning (BERT + TF-IDF)
* 🌍 Real-time web search (Google / News APIs)
* 🤖 AI reasoning (LLM-based judgment)
* ⚖️ Source credibility & consensus analysis

Users can input **text or URL**, and the system returns:

* ✅ REAL / ❌ FAKE / ⚠️ UNCERTAIN
* 📊 Confidence score
* 🔍 Certainty level (LOW / MEDIUM / HIGH)
* 📰 Supporting articles (for real news)

---

## 🧠 Core Idea

> 🔥 “Collect Evidence → Analyze Context → Reason → Decide”

Unlike traditional classifiers, VeritAI acts as a:

👉 **News Fact Checker + AI Judge**

---

## 🗂 Project Structure

```
AI-Based-Fake-News-Detector/
│
├── backend/
│   ├── app.py
│   ├── .env
│   ├── requirements.txt
│   └── __init__.py
│
├── ml_model/
│   ├── model.py
│   ├── tfidf_model.pkl
│   ├── bert_model/
│   └── __init__.py
│
├── frontend/
│   └── index.html
│
└── README.md
```

---

## 🏗 System Architecture

```
Frontend (HTML/JS)
        ↓
Flask API (app.py)
        ↓
AI Engine (model.py)
        ↓
ML Models + APIs + LLM
        ↓
Final Decision
```

---

## 🔁 Data Flow

```
User Input (Text / URL)
        ↓
Translation (if Hindi/Bengali)
        ↓
ML Models (BERT + TF-IDF)
        ↓
Query Generation (LLM)
        ↓
Evidence Retrieval:
    - Google (Serper API)
    - NewsAPI
    - GNews
    - Regional scraping
        ↓
Source Analysis + Semantic Matching
        ↓
Final Decision (REAL / FAKE / UNCERTAIN)
        ↓
Frontend Display
```

---

## 🧠 AI Components

### 🔹 1. BERT Model

* Context understanding
* Detects meaning of news
* Weight: **60%**

---

### 🔹 2. TF-IDF + Logistic Regression

* Pattern detection
* Detects fake-style writing
* Weight: **20%**

---

### 🔹 3. AI Judge (LLM - Groq / LLaMA3)

* Logical reasoning
* Interprets evidence
* Weight: **20%**

---

## 🌍 Evidence Retrieval System

| Source                  | Purpose                      |
| ----------------------- | ---------------------------- |
| **Serper API (Google)** | 🔥 Real-time & regional news |
| **NewsAPI**             | Structured global news       |
| **GNews API**           | Additional coverage          |
| **Web Scraping**        | Bengali / Hindi local news   |

---

## 🌐 Multilingual Support

Supports:

* 🇬🇧 English
* 🇮🇳 Hindi
* 🇧🇩 Bengali

👉 Non-English text is translated using LLM before analysis.

---

## ⚖️ Decision System

### 🔍 Factors Used:

* Source consensus (number of trusted sources)
* Semantic similarity (claim vs evidence)
* Source credibility
* ML predictions (BERT + TF-IDF)

---

### 📊 Output Labels

| Label        | Meaning                          |
| ------------ | -------------------------------- |
| ✅ REAL       | Verified by evidence             |
| ❌ FAKE       | Contradicted or false            |
| ⚠️ UNCERTAIN | Breaking / insufficient evidence |

---

### 📈 Certainty Levels

| Level    | Meaning                       |
| -------- | ----------------------------- |
| HIGH     | Multiple trusted sources      |
| MEDIUM   | Limited but reliable evidence |
| LOW      | Weak evidence but matching    |
| VERY LOW | No strong proof               |

---

## 🚀 Features

* 🧠 AI reasoning-based verification
* 🌍 Real-time news detection
* 📊 Confidence scoring
* ⚖️ Source credibility analysis
* 🌐 Multilingual support
* ⚡ Fast CPU-based inference
* 📰 “Read More” articles for real news

---

## ⚙️ Tech Stack

| Layer    | Technology                        |
| -------- | --------------------------------- |
| Frontend | HTML, CSS, JavaScript             |
| Backend  | Flask, Flask-CORS                 |
| ML/NLP   | scikit-learn, Transformers (BERT) |
| AI       | Groq (LLaMA3)                     |
| APIs     | Serper, NewsAPI, GNews            |
| Parsing  | BeautifulSoup                     |

---

## 🧪 Running the Project

### 1️⃣ Backend + Frontend (served by Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Open:

```txt
http://127.0.0.1:5000/
```

---

## 🔐 Environment Variables (`backend/.env`)

```env
DATABASE_URL=postgresql://user:pass@localhost:5432/veritai
FLASK_SECRET_KEY=change-me

GROQ_API_KEY=your_key
NEWS_API_KEY=your_key
GNEWS_API_KEY=your_key
SERPER_API_KEY=your_key

# OTP email via Brevo
BREVO_API_KEY=your_brevo_api_key
BREVO_SENDER_EMAIL=no-reply@yourdomain.com
BREVO_SENDER_NAME=VeritAI
```

---

## ⚠️ Challenges Solved

| Problem                      | Solution                             |
| ---------------------------- | ------------------------------------ |
| Breaking news detection      | Real-time Google search (Serper)     |
| Regional news (India/Bengal) | Multilingual scraping + query tuning |
| False FAKE predictions       | Semantic matching + flexible logic   |
| API delays                   | Parallel API calls                   |
| Accuracy                     | Hybrid ML + LLM approach             |

---

## 📊 Model Behavior

| Scenario         | Output                               |
| ---------------- | ------------------------------------ |
| Verified news    | REAL                                 |
| Fake viral claim | FAKE                                 |
| Breaking news    | UNCERTAIN → REAL (as evidence grows) |

---

## 🔮 Future Scope

* 📊 Source credibility scoring UI
* 🔍 Highlight fake parts in text
* 📱 Mobile app
* 🌍 More language support
* 🧠 Advanced transformer fine-tuning
* 📡 Live news dashboard

---

## 👨‍💻 Author

**B.Tech Computer Science — Final Year**
AI-Based Fake News Detection System
Academic Year: 2024–25

---

## 📄 License

This project is developed for academic and research purposes.