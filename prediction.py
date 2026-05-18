import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

MODEL_PATH = "model/rf_model.pkl"
TOP_GENRES = [
    "Drama", "Comedy", "Thriller", "Action", "Romance",
    "Horror", "Adventure", "Crime", "Science Fiction", "Mystery",
]
FEATURE_COLS = (
    ["budget", "log_budget", "runtime", "release_year", "genre_count", "keyword_count"]
    + [f"genre_{g.lower().replace(' ', '_')}" for g in TOP_GENRES]
)

def parse_json_list(text):
    try:
        items = json.loads(text.replace("'", '"'))
        return [item["name"] for item in items]
    except:
        return []


def extract_features(df):
    df = df.copy()
    df["genre_count"] = df["genres"].apply(lambda x: len(parse_json_list(x)))
    df["keyword_count"] = df["keywords"].apply(lambda x: len(parse_json_list(x)))
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0)
    median_budget = df[df["budget"] > 0]["budget"].median()
    median_runtime = df[df["runtime"] > 0]["runtime"].median()
    df["budget"] = df["budget"].replace(0, median_budget)
    df["runtime"] = df["runtime"].replace(0, median_runtime)
    df["log_budget"] = np.log1p(df["budget"])
    for genre in TOP_GENRES:
        key = f"genre_{genre.lower().replace(' ', '_')}"
        df[key] = df["genres"].apply(lambda x, g=genre: 1 if g in parse_json_list(x) else 0)
    return df


def train_model(df):
    df = extract_features(df)
    df = df.dropna(subset=FEATURE_COLS + ["popularity"])
    X = df[FEATURE_COLS]
    y = np.log1p(df["popularity"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Popularity Model — R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLS, "top_genres": TOP_GENRES}, f)
    print(f"Model saved to {MODEL_PATH}")
    return model, {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)}


def load_model():
    data = pickle.load(open(MODEL_PATH, "rb"))
    if isinstance(data, dict):
        return data["model"]
    return data


def predict(model, budget, runtime, release_year, genre_count, keyword_count, genre_list=None):
    if genre_list is None:
        genre_list = []
    row = {"budget": budget, "log_budget": np.log1p(budget), "runtime": runtime,
           "release_year": release_year, "genre_count": genre_count, "keyword_count": keyword_count}
    for genre in TOP_GENRES:
        row[f"genre_{genre.lower().replace(' ', '_')}"] = 1 if genre in genre_list else 0
    input_df = pd.DataFrame([row])
    pred_log = model.predict(input_df)[0]
    popularity = round(float(np.expm1(pred_log)), 1)
    if popularity >= 10.0:
        status, status_class = "Hit", "hit"
    elif popularity >= 2.0:
        status, status_class = "Average", "average"
    else:
        status, status_class = "Flop", "flop"
    return {"popularity": popularity, "status": status, "status_class": status_class}
