import os
from flask import Flask, render_template, request, jsonify
from recommendation import (
    load_data,
    load_posters,
    preprocess,
    build_similarity_matrix,
    recommend,
    get_trending,
)
from prediction import train_model, load_model, predict as predict_success

app = Flask(__name__)

# Recommendation system dataset (needs preprocessing)
print("Loading dataset for recommendations...")
df = load_data()
df = preprocess(df)
print(f"Loaded {len(df)} movies")
similarity = build_similarity_matrix(df)
print("Similarity matrix built")

poster_map = load_posters()
print(f"Poster cache: {len(poster_map)} movies with posters")

# Load or train prediction model (uses raw data, no preprocessing)
print("Loading prediction model...")
if os.path.exists("model/rf_model.pkl"):
    pred_model = load_model()
    print("Prediction model loaded from pickle")
else:
    print("Training prediction model...")
    raw_df = load_data()
    pred_model, pred_metrics = train_model(raw_df)
    print(f"Prediction model trained (R²={pred_metrics['r2']})")


# Homepage — search hero + trending movies grid
@app.route("/")
def index():
    trending = get_trending(df, 20, poster_map)
    return render_template("index.html", trending=trending)


# Autocomplete API — returns matching movie titles as JSON (max 10)
@app.route("/api/search")
def search():
    q = request.args.get("q", "").lower()
    if not q:
        return jsonify([])
    matches = df[df["title"].str.lower().str.contains(q, na=False)]["title"].head(10).tolist()
    return jsonify(matches)


# Core ML endpoint — receives movie title, returns 10 similar movies
@app.route("/recommend", methods=["POST"])
def recommend_route():
    data = request.get_json()
    title = data.get("title", "")
    results = recommend(title, df, similarity, 10, poster_map)
    return jsonify({"recommendations": results, "movie": title})


# Movie detail page with poster + info + More Like This recommendations
@app.route("/movie/<path:title>")
def movie_page(title):
    match = df[df["title"] == title]
    if match.empty:
        return render_template("index.html", trending=get_trending(df, 20, poster_map), error="Movie not found")
    movie = match.iloc[0]
    poster_url = None
    if poster_map and movie["id"] in poster_map:
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_map[movie['id']]}"
    recs = recommend(title, df, similarity, 10, poster_map)
    return render_template(
        "movie.html",
        movie={
            "title": movie["title"],
            "genres": movie["genres"],
            "overview": movie["overview"],
            "vote_average": round(movie["vote_average"], 1),
            "popularity": round(movie["popularity"], 1),
            "tagline": movie.get("tagline", ""),
            "poster_url": poster_url,
        },
        recommendations=recs,
    )


# Prediction form page
@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# Predict API — receives movie features, returns predicted popularity + status
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    result = predict_success(
        pred_model,
        budget=data.get("budget", 0),
        runtime=data.get("runtime", 0),
        release_year=data.get("release_year", 2025),
        genre_count=data.get("genre_count", 0),
        keyword_count=data.get("keyword_count", 0),
        genre_list=data.get("genres", []),
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
