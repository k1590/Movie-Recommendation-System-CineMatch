from flask import Flask, render_template, request, jsonify
from recommendation import (
    load_data,
    load_posters,
    preprocess,
    build_similarity_matrix,
    recommend,
    get_trending,
)

app = Flask(__name__)

# Load dataset and build similarity matrix once at server startup
print("Loading dataset...")
df = load_data()
df = preprocess(df)
print(f"Loaded {len(df)} movies")
similarity = build_similarity_matrix(df)
print("Similarity matrix built")

poster_map = load_posters()
print(f"Poster cache: {len(poster_map)} movies with posters")


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


if __name__ == "__main__":
    app.run(debug=True)
