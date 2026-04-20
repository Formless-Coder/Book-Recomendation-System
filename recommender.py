"""
Book Recommendation System — Core Module
KNN-based recommendation using NearestNeighbors (Ball Tree).
"""

import numpy as np
import pandas as pd
import time
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler

# ─── Global state ────────────────────────────────────────────────────────────
_df           = None   # raw dataframe
_df2          = None   # processed dataframe
_titles_lower = None   # pre-calculated lowercase titles
_idlist       = None   # neighbour index list
_model        = None   # fitted KNN model
_is_ready     = False


def load_and_train(csv_path: str = "books.csv") -> dict:
    """
    Load the dataset, build features, and fit the KNN model.
    Returns a status dict  {ok: bool, message: str, book_count: int}.
    """
    global _df, _df2, _titles_lower, _idlist, _model, _is_ready

    try:
        start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Core: Loading dataset...")
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
        df.dropna(subset=["average_rating", "ratings_count", "language_code", "title", "authors"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df2 = df.copy()
        # Pre-calculating lowercase titles for faster search
        _titles_lower = df2["title"].str.lower().values

        print(f"[{time.strftime('%H:%M:%S')}] Core: Building features...")
        # Rating buckets
        bins   = [0, 1, 2, 3, 4, 5]
        labels = ["0-1", "1-2", "2-3", "3-4", "4-5"]
        df2["rating_between"] = pd.cut(df2["average_rating"], bins=bins, labels=labels, include_lowest=True)

        rating_df   = pd.get_dummies(df2["rating_between"], prefix="rating")
        language_df = pd.get_dummies(df2["language_code"], prefix="lang")

        features_raw = pd.concat(
            [rating_df, language_df, df2["average_rating"], df2["ratings_count"]],
            axis=1,
        )
        features_raw.columns = features_raw.columns.astype(str)

        scaler   = MinMaxScaler()
        features = scaler.fit_transform(features_raw)

        print(f"[{time.strftime('%H:%M:%S')}] Core: Training KNN model...")
        model = neighbors.NearestNeighbors(n_neighbors=6, algorithm="ball_tree")
        model.fit(features)
        _, idlist = model.kneighbors(features)

        _df, _df2, _idlist, _model = df, df2, idlist, model
        _is_ready = True
        
        duration = time.time() - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Core: Initialization complete in {duration:.2f}s")

        return {"ok": True, "message": "Model ready!", "book_count": len(df)}

    except FileNotFoundError:
        return {"ok": False, "message": f"Dataset not found: {csv_path}", "book_count": 0}
    except Exception as e:
        return {"ok": False, "message": str(e), "book_count": 0}


def get_all_titles() -> list[str]:
    """Return sorted list of all book titles (for autocomplete)."""
    if _df2 is None:
        return []
    return sorted(_df2["title"].unique().tolist())


def recommend(book_name: str, n: int = 5) -> list[dict]:
    """
    Return up to `n` recommendations (excluding the query book itself).
    Each dict has: title, authors, average_rating, ratings_count, language_code.
    """
    if not _is_ready:
        raise RuntimeError("Model not loaded. Call load_and_train() first.")

    start_time = time.time()
    query = book_name.strip().lower()
    print(f"[{time.strftime('%H:%M:%S')}] Engine: Searching for '{query}'...")

    # 1. Exact match search using pre-calculated index
    matches = _df2[_titles_lower == query]
    
    # 2. Fuzzy fallback
    if matches.empty:
        print(f"[{time.strftime('%H:%M:%S')}] Engine: No exact match, trying fuzzy fallback...")
        # Using regex=False for speed and safety
        matches = _df2[_df2["title"].str.lower().str.contains(query, na=False, regex=False)]

    if matches.empty:
        print(f"[{time.strftime('%H:%M:%S')}] Engine: No matches found.")
        return []

    print(f"[{time.strftime('%H:%M:%S')}] Engine: Found {len(matches)} matches. Retrieving neighbors...")
    book_id    = matches.index[0]
    neighbours = _idlist[book_id]

    results = []
    for nid in neighbours:
        row = _df2.loc[nid]
        if str(row["title"]).lower() == query:
            continue
        results.append({
            "title":         str(row["title"]),
            "authors":       str(row["authors"]),
            "average_rating": round(float(row["average_rating"]), 2),
            "ratings_count": int(row["ratings_count"]),
            "language_code": str(row["language_code"]),
        })
        if len(results) >= n:
            break

    duration = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] Engine: Recommendation completed in {duration:.4f}s")
    return results
