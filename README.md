# 🧠 Integrated Sentiment Analysis for Disaster Tweets (Streamlit)

🔗 **Deployment Page:**  
https://analyticappcialmedia-sentiment-app-fireforest.streamlit.app/

This Streamlit application demonstrates **sentiment analysis of Twitter data**
related to **forest fires (*kebakaran hutan*) and haze (*kabut asap*)** using a
transformer-based model:  
`cardiffnlp/twitter-xlm-roberta-base-sentiment`.

The app is part of an academic project that integrates **Sentiment Analysis**
and **Social Network Analysis (SNA)** for disaster communication research.

---

## 📦 Features

- ✍️ Tweet text input
- 🤖 Sentiment classification:
  - Positive
  - Neutral
  - Negative
- 🔥 Disaster-related keyword detection
- 📊 Project overview aligned with research paper
- 🖥️ Interactive UI built with Streamlit

> ⚠️ Note: Sentiment prediction reflects **emotional polarity**, not factual correctness.

---

## 🤖 Model

- **Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- **Architecture:** XLM-RoBERTa (Transformer-based)
- **Type:** Multilingual (Indonesian & English)
- **Inference:** Zero-shot (no additional fine-tuning)

---

## 📂 Dataset (Research Pipeline)

The dataset was collected using a **custom Twitter scraping framework** that retrieves:
- Root tweets
- Replies
- Conversation threads
- Quoted tweets

**Dataset statistics:**
- Total tweets: 4,199
- Unique users: 3,609
- Conversation threads: 1,305
- Interaction edges: 4,199

Each tweet record includes:
- `tweet_id`
- `parent_tweet_id`
- `username`
- `tweet_text`
- `sentiment`
- `sentiment_score`
- `user_sentiment_score`

---

## 🚀 How to Run Locally

### 1️⃣ Clone Repository and Enter Folder
```bash
git clone https://github.com/laelacitrasih/SentimentAnalysis
cd SentimentAnalysis
```

### 2. Virtual Environment Activation
```bash
python3 -m venv env
source env/bin/activate  # Mac/Linux
# atau
env\Scripts\activate     # Windows
```

### 3. Install Dependency
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```
