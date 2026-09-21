import sqlite3
import math
import numpy as np
import requests
from bs4 import BeautifulSoup
from collections import Counter
from scipy.sparse import csr_matrix
import joblib

# import STOPWORDS and scraper directly from Scraper.py
# this ensures training and inference use identical tokenisation
from Scraper import STOPWORDS, scraper

import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "tf_idf.db")
MODEL_PATH = os.path.join(BASE_DIR, "bias_regression_model.joblib")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab_wordid_to_col.joblib")


def build_vector(tokens, total_words, db_path, wordid_to_col):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # get total article count for IDF calculation
    cur.execute("SELECT COUNT(*) FROM TBLarticles;")
    N = cur.fetchone()[0]

    # count how many times each token appears in this article
    counts = Counter(tokens)

    # three parallel lists for sparse matrix construction
    row_ind = []
    col_ind = []
    data = []

    for word, count in counts.items():
        # check if  word exists in training vocab
        cur.execute(
            "SELECT word_id FROM TBLdictionary WHERE word = ?;",
            (word,)
        )
        result = cur.fetchone()
        if result is None:
            continue

        word_id = result[0]

        # check if this word survived pruning 
        # wordid_to_col only contains words that passed min_df and max_df
        if word_id not in wordid_to_col:
            continue

        col_index = wordid_to_col[word_id]

        # get document frequency for IDF calculation
        cur.execute(
            "SELECT df FROM TBLdocumentfrequency WHERE word_id = ?;",
            (word_id,)
        )
        df_row = cur.fetchone()
        if df_row is None:
            continue

        df = df_row[0]

        # calculate TF-IDF 
        tf = count / total_words
        idf = math.log((N + 1) / (df + 1)) + 1.0
        tfidf = tf * idf

        # all values go in row 0 because this is a single article
        row_ind.append(0)
        col_ind.append(col_index)
        data.append(tfidf)

    conn.close()

    # build a single row sparse matrix of shape (1, vocab_size)
    X = csr_matrix((data, (row_ind, col_ind)), shape=(1, len(wordid_to_col)))
    return X


def scale_to_0_100(pred):
    # ridge regression is unbounded so raw predictions can exceed 0-100
    # clip to valid range before displaying to user
    return np.clip(pred, 0.0, 100.0)


def main():
    url = input("Enter URL to test: ").strip()

    print("Scraping...")
    tokens, total_words = scraper(url)

    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    wordid_to_col = joblib.load(VOCAB_PATH)

    print("Building vector...")
    X = build_vector(tokens, total_words, DB_PATH, wordid_to_col)

    print("Predicting...")
    pred_raw = model.predict(X)[0]
    pred_scaled = scale_to_0_100(pred_raw)

    print("\n--- RESULT ---")
    print(f"Scaled score (0-100): {round(pred_scaled, 2)}")
    print("0 = Left | 50 = Uncertain | 100 = Right")


if __name__ == "__main__":
    main()