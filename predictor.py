import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# =========================================================
# Model & Tokenizer Loader
# =========================================================
@torch.no_grad()
def load_model_and_tokenizer():
    """
    Load multilingual sentiment analysis model
    aligned with the research paper.
    """
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # id2label mapping from model config
    label_dict = model.config.id2label

    return model, tokenizer, label_dict


# =========================================================
# Single Tweet Sentiment Prediction
# =========================================================
@torch.no_grad()
def predict_sentiment(text, model, tokenizer, label_dict):
    """
    Perform sentiment inference on a single tweet.
    Returns a human-readable sentiment label.
    """

    encoded_input = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model(**encoded_input)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    raw_label = label_dict[pred_id]

    # Normalize labels (model-dependent)
    label_map = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
    }

    return label_map.get(raw_label.lower(), raw_label)


# =========================================================
# Batch Sentiment Prediction (Optional but Useful)
# =========================================================
@torch.no_grad()
def predict_sentiment_batch(texts, model, tokenizer, label_dict):
    """
    Perform sentiment inference on a list of tweets.
    Returns a list of sentiment labels.
    """

    encoded_input = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = model(**encoded_input)
    pred_ids = torch.argmax(outputs.logits, dim=1).tolist()

    sentiments = []
    for pid in pred_ids:
        raw_label = label_dict[pid]
        sentiments.append(
            {
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive",
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive",
            }.get(raw_label.lower(), raw_label)
        )

    return sentiments


# =========================================================
# Load TweetEval Dataset (Demo / Reference)
# =========================================================
def load_tweet_dataset():
    """
    Load TweetEval sentiment dataset
    (used for demonstration and exploration).
    """
    return load_dataset("cardiffnlp/tweet_eval", "sentiment")


# =========================================================
# Label ID → Name Mapping
# =========================================================
def get_label_map():
    return {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
