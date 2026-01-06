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

    tab1, tab2, tab3 = st.tabs(
        ["🧭 Overview", "📊 Data & Model", "📄 Paper & Slides"]
    )

    # -----------------------------------------------------
    # TAB 1: OVERVIEW
    # -----------------------------------------------------
    with tab1:
        st.markdown("""
        ### 🧠 Research Context

        Forest fire (*kebakaran hutan*) and haze (*kabut asap*) disasters are recurring
        environmental crises that generate strong public reactions on social media.
        Twitter enables users to rapidly share information, express emotions, and
        interact during disaster events.

        This project analyzes **public sentiment and interaction patterns** related to
        forest fire and haze disasters using **Twitter data**, with a focus on
        understanding how emotions propagate through online networks.

        ---
        ### 🎯 Objectives (Aligned with Course & Paper)

        This project is designed to enable students and researchers to:

        1. **Classify public opinions and emotions** related to disaster events  
        2. **Integrate sentiment analysis with social network analysis (SNA)**  
        3. **Identify influential actors and discussion communities**  
        4. **Generate insights relevant to disaster communication and mitigation**

        ---
        ### 🔍 System Scope

        - Tweet-level sentiment inference (interactive demo)
        - Disaster-related keyword detection
        - Offline analytical pipeline:
          - Social Network Construction
          - Community Detection
          - Centrality Analysis
          - Integrated Sentiment–Network Analysis

        The Streamlit application serves as a **demonstration interface**, while the
        complete analytical results are reported in the research paper.
        """)

    # -----------------------------------------------------
    # TAB 2: DATA & MODEL
    # -----------------------------------------------------
    with tab2:
        st.markdown("### 📂 Dataset Description")

        st.markdown("""
        The dataset was collected using a **custom Twitter scraping framework** that
        retrieves multiple forms of user interaction to support sentiment and network
        analysis, including:

        - Root tweets  
        - Replies  
        - Conversation threads  
        - Quoted tweets  

        ---
        ### 📊 Dataset Statistics

        - **Total tweets:** 4,199  
        - **Unique users:** 3,609  
        - **Conversation threads:** 1,305  
        - **Interaction edges:** 4,199  

        ---
        ### 🗃️ Tweet Record Structure

        Each tweet record includes the following attributes:

        - `tweet_id`  
        - `parent_tweet_id`  
        - `username`  
        - `tweet_text`  
        - `sentiment`  
        - `sentiment_score`  
        - `user_sentiment_score`  

        These fields enable **tweet-level**, **user-level**, and
        **network-level** analysis.
        """)

        st.markdown("### 🤖 Sentiment Analysis Model")

        st.markdown("""
        - **Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`
        - **Architecture:** XLM-RoBERTa (Transformer-based)
        - **Training Data:** Twitter-based sentiment benchmark
        - **Inference Type:** Zero-shot (no additional fine-tuning)
        """)

        st.markdown("""
        ### 🌐 Model Capabilities

        - Multilingual support (Indonesian & English)
        - Robust to informal and noisy Twitter language
        - Suitable for real-time sentiment inference
        - Outputs **three sentiment classes**:
          - Positive
          - Neutral
          - Negative
        """)

        st.markdown("""
        ### 📄 Relation to Research Paper

        This Streamlit application implements the **Sentiment Analysis Module**
        described in the paper:

        > *Integrated Sentiment Analysis and Social Network Analysis for Forest Fire and Haze Disasters*

        Advanced analyses—including network construction, community detection,
        centrality analysis, and sentiment–network integration—are performed
        offline and reported in the paper.
        """)

        st.info(
            "ℹ️ Note: Sentiment predictions reflect emotional polarity, "
            "not factual correctness or ground truth validation."
        )

    # -----------------------------------------------------
    # TAB 3: PAPER & SLIDES
    # -----------------------------------------------------
    with tab3:
        st.subheader("📄 Final Paper & Presentation")

        st.markdown("""
        This project is documented in an **IEEE-format research paper** and
        accompanied by presentation slides.

        ### 📝 Resources

        - 📄 **Paper (IEEE Format)**  
          https://www.overleaf.com/read/jgvvvppjmwyk#119cf3

        - 📊 **Presentation Slides**  
          https://www.canva.com/design/DAG9qyPyaPk/_vTaj5GyB6tvLdyJH1lYQw/edit?utm_content=DAG9qyPyaPk&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

        ---
        These materials provide a complete explanation of:
        - Methodology
        - Experimental results
        - Integrated sentiment–network analysis
        - Implications for disaster management
        """)
