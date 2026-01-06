import streamlit as st
import pandas as pd
from predictor import load_model_and_tokenizer, predict_sentiment

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Disaster Tweet Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Session State
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# =========================================================
# Sidebar Navigation
# =========================================================
with st.sidebar:
    st.markdown("## 📌 Menu")
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
    if st.button("📊 Project Overview"):
        st.session_state.page = "Project Overview"

# =========================================================
# Load Model
# =========================================================
model, tokenizer, label_names = load_model_and_tokenizer()

# =========================================================
# Disaster Keywords (Aligned with Paper)
# =========================================================
DISASTER_KEYWORDS = [
    "kebakaran hutan", "kabut asap", "karhutla", "hutan terbakar",
    "api hutan", "asap", "titik api", "hotspot",
    "forest fire", "wildfire", "haze", "smog"
]

# =========================================================
# HOME PAGE
# =========================================================
if st.session_state.page == "Home":

    st.title("🔥 Disaster Tweet Sentiment Analysis")
    st.markdown("""
    This demo illustrates the **Sentiment Analysis Module** proposed in the project  
    *“Integrated Sentiment Analysis and Social Network Analysis for Forest Fire and Haze Disasters.”*
    """)

    st.info(
        "This application demonstrates **tweet-level sentiment inference**. "
        "Network and community analysis are performed offline as described in the paper."
    )

    tweet = st.text_area(
        "✍️ Enter a tweet (Indonesian or English):",
        placeholder="Contoh: Kabut asap makin parah, pemerintah harus bertindak!"
    )

    if st.button("🔍 Analyze Sentiment"):

        if tweet.strip():

            sentiment = predict_sentiment(tweet, model, tokenizer, label_names)
            st.success(f"**Predicted Sentiment:** {sentiment}")

            # Keyword detection
            tweet_lower = tweet.lower()
            detected = [k for k in DISASTER_KEYWORDS if k in tweet_lower]

            if detected:
                st.warning("⚠️ Disaster-related content detected")
                st.write("**Keywords found:**", ", ".join(detected))
            else:
                st.info("ℹ️ No disaster-related keywords detected")

        else:
            st.warning("Please enter a tweet.")

# =========================================================
# PROJECT OVERVIEW PAGE
# =========================================================
elif st.session_state.page == "Project Overview":

    st.title("📊 Project Overview")

    tab1, tab2, tab3 = st.tabs(["Overview", "Data & Model", "Paper & Slides"])

    # -----------------------------------------------------
    # TAB 1: OVERVIEW
    # -----------------------------------------------------
    with tab1:
        st.markdown("""
        ### Research Context

        This project analyzes **public sentiment and interaction patterns** related to:
        - 🔥 Forest fires (*kebakaran hutan*)
        - 🌫️ Haze disasters (*kabut asap*)

        using **Twitter data**.

        ### Objectives (Aligned with Paper)
        - Classify public sentiment (positive, negative, neutral)
        - Understand emotional responses during disasters
        - Support disaster communication and mitigation strategies

        ### System Scope
        - Tweet-level sentiment inference (demo)
        - Network and community analysis (offline, research pipeline)
        """)

    # -----------------------------------------------------
    # TAB 2: DATA & MODEL
    # -----------------------------------------------------
    with tab2:
        st.markdown("### 🤖 Sentiment Analysis Model")

        st.markdown("""
        - **Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
        - **Architecture:** XLM-RoBERTa (Transformer-based)
        - **Capabilities:**  
          - Multilingual (Indonesian & English)
          - Robust to informal Twitter language
        - **Output:** Positive / Neutral / Negative
        """)

        st.markdown("""
        ### 📌 Notes on Usage
        - The model performs **zero-shot inference**
        - No additional fine-tuning was applied
        - Predictions reflect **tweet-level sentiment**, not factual correctness
        """)

        st.markdown("""
        ### 📄 Relation to Research Paper
        This application implements the **Sentiment Analysis Module** described in:
        > *Integrated Sentiment Analysis and Social Network Analysis for Forest Fire and Haze Disasters*

        Network construction, community detection, and centrality analysis are conducted
        using offline pipelines and are reported in the paper.
        """)

    with tab3:
            st.subheader("📄 Final Paper")
            st.markdown("Berikut adalah paper penelitian yang ditulis dalam format IEEE berdasarkan proyek ini:")
        
            st.markdown("""
            ### 📝 Paper & Slides
        
            - Paper IEEE Format: [Sharelatex](https://www.overleaf.com/read/jgvvvppjmwyk#119cf3)
            - Slides: [Canva](https://www.canva.com/design/DAGlgd6NJcs/13a6Ezvop1ocBAKt0I2WNg/edit?utm_content=DAGlgd6NJcs&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
        
            """)