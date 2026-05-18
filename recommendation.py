import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

IMG_BASE = "https://image.tmdb.org/t/p/w500"


# Parse JSON list like '[{"name":"Action"}]' into ["Action"]
def parse_json_list(text):
    try:
        items = json.loads(text.replace("'", '"'))
        return [item["name"] for item in items]
    except:
        return []


# Load TMDB movie dataset from CSV
def load_data(movies_path="dataset/tmdb_5000_movies.csv"):
    df = pd.read_csv(movies_path)
    return df


# Load cached poster paths into dict {movie_id: poster_path}
def load_posters(cache_path="dataset/posters_cache.csv"):
    try:
        posters = pd.read_csv(cache_path)
        return dict(zip(posters["id"], posters["poster_path"]))
    except:
        return {}


# Merge genres + keywords + overview into a single "tags" column
def preprocess(df):
    df["genres"] = df["genres"].apply(parse_json_list)
    df["keywords"] = df["keywords"].apply(parse_json_list)

    df["tags"] = df.apply(
        lambda row: " ".join(
            [g.lower().replace(" ", "") for g in row["genres"]]
            + [k.lower().replace(" ", "") for k in row["keywords"]]
            + [str(row["overview"]).lower() if pd.notna(row["overview"]) else ""]
        ),
        axis=1,
    )
    return df


# Convert tags to vectors with CountVectorizer, then compute cosine similarity matrix
def build_similarity_matrix(df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity


# Return top N trending movies sorted by TMDB popularity score
def get_trending(df, n=20, poster_map=None):
    top = df.sort_values("popularity", ascending=False).head(n)
    results = []
    for _, row in top.iterrows():
        poster_url = None
        if poster_map and row["id"] in poster_map:
            poster_url = f"{IMG_BASE}{poster_map[row['id']]}"
        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "overview": row["overview"],
            "vote_average": round(row["vote_average"], 1),
            "popularity": round(row["popularity"], 1),
            "poster_url": poster_url,
        })
    return results


# Find top N similar movies using the pre-computed cosine similarity matrix
def recommend(movie_title, df, similarity, n=10, poster_map=None):
    if movie_title not in df["title"].values:
        return []

    idx = df[df["title"] == movie_title].index[0]
    # Get similarity scores, sort descending, skip self (score=1 at index 0)
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : n + 1]

    results = []
    for i, score in scores:
        row = df.iloc[i]
        poster_url = None
        if poster_map and row["id"] in poster_map:
            poster_url = f"{IMG_BASE}{poster_map[row['id']]}"
        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "overview": row["overview"][:200] + "..."
            if pd.notna(row["overview"]) and len(str(row["overview"])) > 200
            else row["overview"],
            "vote_average": round(row["vote_average"], 1),
            "similarity": round(score * 100, 1),
            "popularity": round(row["popularity"], 1),
            "poster_url": poster_url,
        })
    return results
