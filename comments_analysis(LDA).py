"""
YouTube Comment Analysis
========================

A Python script for analyzing YouTube comments using NLP, sentiment analysis,
and topic modeling.

Input:
    data/youtube_comments.csv  (must contain "Comment" and "Level" columns)

Output:
    - youtube_comments_with_wordcount.csv
    - youtube_comments_with_upper_flag.csv
    - youtube_comments_with_pos.csv
    - youtube_pos_counts.csv
    - youtube_comments_with_sentiment.csv
    - word_frequency_youtube.csv
    - youtube_lda_bubbles.html

Contact: florencewu103@gmail.com
"""

import os
import re
import json
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from collections import Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim import corpora
from gensim.models import CoherenceModel

from tqdm import tqdm

import pyLDAvis


# ============================================================
# Configuration
# ============================================================

# Path to the input CSV file
# The file should contain at least two columns: "Comment" and "Level"
INPUT_PATH = "data/youtube_comments.csv"

# Words that should be kept even if they appear in the stopword list
PRESERVE_WORDS = {"love and deepspace", "caleb"}

# Custom stopwords specific to this corpus
CUSTOM_STOPWORDS = {
    "im", "ok", "so", "need", "get", "like", "going", "want",
    "got", "okay", "gon", "na", "ca", "cant",
}

# Range of topic numbers to evaluate
K_RANGE = range(2, 11)

# Final number of topics (selected after coherence/perplexity evaluation)
BEST_K = 3

# Random seed for reproducibility
RANDOM_STATE = 42


# ============================================================
# Step 1: Load and Clean Data
# ============================================================

def load_and_clean(path):
    print("[1/8] Loading and cleaning data...")

    df = pd.read_csv(path)
    df["Comment"] = df["Comment"].astype(str)

    # Keep only comments containing English letters
    df = df[df["Comment"].str.contains(r"[A-Za-z]")].reset_index(drop=True)

    # Keep only top-level comments (L1)
    df = df[df["Level"] == "L1"].reset_index(drop=True)

    print(f"    Loaded {len(df)} top-level English comments")
    return df


# ============================================================
# Step 2: Word Count and All-Caps Detection
# ============================================================

def add_word_count_and_upper_flag(df):
    print("[2/8] Counting words and detecting all-caps comments...")

    # Count words per comment
    df["word_count"] = df["Comment"].apply(
        lambda txt: len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", txt))
    )

    # Detect comments written entirely in uppercase
    def is_all_upper(text):
        letters = re.sub(r"[^A-Za-z]", "", text)
        return letters.isupper() and letters != ""

    df["is_upper"] = df["Comment"].apply(is_all_upper)

    upper_count = df["is_upper"].sum()
    total = len(df)
    ratio = upper_count / total if total else 0

    print(f"    All-caps comments: {upper_count}")
    print(f"    Proportion of total: {ratio:.2%}")

    df.to_csv("youtube_comments_with_wordcount.csv",
              index=False, encoding="utf-8-sig")
    df.to_csv("youtube_comments_with_upper_flag.csv",
              index=False, encoding="utf-8-sig")
    return df


# ============================================================
# Step 3: POS Tagging
# ============================================================

def pos_tagging(df):
    print("[3/8] Running POS tagging...")

    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    def pos_tag_comment(text):
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return json.dumps(tagged, ensure_ascii=False)

    df["Comment_POS"] = df["Comment"].apply(pos_tag_comment)

    df.to_csv("youtube_comments_with_pos.csv",
              index=False, encoding="utf-8-sig")

    # Extract and count POS tags
    def extract_tags(cell):
        if isinstance(cell, list):
            return [tag for _, tag in cell]
        if isinstance(cell, str):
            try:
                pairs = json.loads(cell)
            except json.JSONDecodeError:
                pairs = ast.literal_eval(cell)
            return [tag for _, tag in pairs if len(_) == 2]
        return []

    df["__tags"] = df["Comment_POS"].apply(extract_tags)

    pos_counts = (
        df["__tags"]
        .explode()
        .value_counts()
        .rename_axis("POS")
        .reset_index(name="Count")
    )
    pos_counts.to_csv("youtube_pos_counts.csv", index=False)

    print(f"    POS tag counts saved (top 5):")
    print(pos_counts.head())
    return df


# ============================================================
# Step 4: Tokenization and Stopword Removal
# ============================================================

def tokenize_and_filter(df):
    print("[4/8] Tokenizing and removing stopwords...")

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Remove @username mentions
    df["Comment"] = df["Comment"].str.replace(r"^@\S+\s*", "", regex=True)
    df = df[df["Comment"].str.contains(r"[A-Za-z0-9]")].reset_index(drop=True)

    default_stop = set(stopwords.words("english"))
    all_stop = default_stop | CUSTOM_STOPWORDS

    def tokenize(text):
        toks = nltk.word_tokenize(text.lower())
        return [t for t in toks if (t.isalpha() or t in PRESERVE_WORDS)]

    df["tokens_raw"] = df["Comment"].apply(tokenize)
    df["tokens"] = df["tokens_raw"].apply(
        lambda toks: [t for t in toks
                      if (t not in all_stop) or (t in PRESERVE_WORDS)]
    )

    print(f"    Tokenization complete ({len(df)} comments)")
    return df


# ============================================================
# Step 5: Sentiment Analysis (VADER)
# ============================================================

def sentiment_analysis(df):
    print("[5/8] Running sentiment analysis (VADER)...")

    nltk.download("vader_lexicon", quiet=True)
    sid = SentimentIntensityAnalyzer()

    df["sent_score"] = df["Comment"].apply(
        lambda x: sid.polarity_scores(x)["compound"]
    )

    def label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sent_label"] = df["sent_score"].apply(label)

    print("    Sentiment distribution:")
    print(df["sent_label"].value_counts())

    df.to_csv("youtube_comments_with_sentiment.csv",
              index=False, encoding="utf-8-sig")
    return df


# ============================================================
# Step 6: Word Frequency and Word Cloud
# ============================================================

def word_frequency_and_cloud(df):
    print("[6/8] Computing word frequency and generating word cloud...")

    all_tokens = sum(df["tokens"], [])
    freq = Counter(all_tokens)

    pd.DataFrame(freq.items(), columns=["token", "count"]) \
        .sort_values("count", ascending=False) \
        .to_csv("word_frequency_youtube.csv", index=False, encoding="utf-8-sig")

    wc = WordCloud(width=800, height=400, background_color="white") \
        .generate_from_frequencies(freq)

    plt.figure(figsize=(10, 4))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("YouTube Comments Word Cloud")
    plt.tight_layout()
    plt.savefig("wordcloud.png", dpi=150)
    plt.close()

    print("    Word cloud saved as wordcloud.png")
    return freq


# ============================================================
# Step 7: Topic Modeling (LDA)
# ============================================================

def topic_modeling(df):
    print("[7/8] Running LDA topic modeling...")

    texts_token = df["tokens"].tolist()

    vectorizer = CountVectorizer(
        tokenizer=lambda x: x, lowercase=False, min_df=2, max_df=0.95
    )
    dtm = vectorizer.fit_transform(texts_token)

    dictionary = corpora.Dictionary(texts_token)

    # Evaluate coherence across different K values
    coh_vals = []
    for k in tqdm(K_RANGE, desc="    Coherence"):
        lda_tmp = LatentDirichletAllocation(
            n_components=k, random_state=RANDOM_STATE
        ).fit(dtm)
        terms = vectorizer.get_feature_names_out()
        topics = [
            [terms[i] for i in comp.argsort()[-15:]]
            for comp in lda_tmp.components_
        ]
        coh = CoherenceModel(
            topics=topics, texts=texts_token,
            dictionary=dictionary, coherence="c_v"
        ).get_coherence()
        coh_vals.append(coh)

    # Plot coherence
    plt.figure(figsize=(6, 4))
    plt.plot(list(K_RANGE), coh_vals, marker="o")
    plt.title("Coherence vs. K")
    plt.xlabel("K")
    plt.ylabel("Coherence (c_v)")
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    plt.savefig("coherence_vs_k.png", dpi=150)
    plt.close()

    # Train final LDA model
    lda = LatentDirichletAllocation(
        n_components=BEST_K, random_state=RANDOM_STATE
    ).fit(dtm)

    terms = vectorizer.get_feature_names_out()
    print(f"    Final topics (K={BEST_K}):")
    for idx, comp in enumerate(lda.components_):
        prob = comp / comp.sum()
        top = prob.argsort()[-12:][::-1]
        line = ", ".join(f"{prob[i]:.3f}*{terms[i]}" for i in top)
        print(f"    Topic {idx}: {line}")

    # Save pyLDAvis visualization
    topic_term = lda.components_ / lda.components_.sum(axis=1)[:, None]
    doc_topic = lda.transform(dtm)
    doc_lengths = np.asarray(dtm.sum(axis=1)).flatten()
    vocab = vectorizer.get_feature_names_out()
    term_freq = np.asarray(dtm.sum(axis=0)).flatten()

    panel = pyLDAvis.prepare(
        topic_term, doc_topic, doc_lengths,
        vocab, term_freq, sort_topics=False
    )
    pyLDAvis.save_html(panel, "youtube_lda_bubbles.html")
    print("    Saved: youtube_lda_bubbles.html")

    return lda, vectorizer, dtm


# ============================================================
# Main
# ============================================================

def main():
    # Check input file
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            f"Please place your CSV file at this location, "
            f"or update INPUT_PATH in the script."
        )

    # Run the pipeline
    df = load_and_clean(INPUT_PATH)
    df = add_word_count_and_upper_flag(df)
    df = pos_tagging(df)
    df = tokenize_and_filter(df)
    df = sentiment_analysis(df)
    word_frequency_and_cloud(df)
    topic_modeling(df)

    print("[8/8] Pipeline complete.")


if __name__ == "__main__":
    main()