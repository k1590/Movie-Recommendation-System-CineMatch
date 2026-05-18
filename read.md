# CineMatch

CineMatch is a web-based Movie Recommendation System built with Python Flask and a content-based recommendation engine.

## What it does
- Recommends similar movies based on a selected movie
- Uses genres, keywords, and plot overview to compute similarity
- Uses TMDB 5000 movie dataset and poster images from TMDB

## Tech stack
- Frontend: HTML, CSS, JavaScript
- Backend: Python Flask
- Data processing: pandas, numpy
- ML: scikit-learn (CountVectorizer + Cosine Similarity)

## Key files
- `app.py` — Flask server and routes
- `recommendation.py` — data processing and recommendation logic
- `fetch_posters.py` — fetches TMDB poster paths and caches them
- `templates/` — HTML templates for pages
- `static/` — CSS and JS assets
- `dataset/` — movie data and poster cache

## How to run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Flask app:
   ```bash
   python app.py
   ```
3. Open the URL shown in the terminal.

## Notes
- The recommendation engine uses a content-based approach, so it finds movies similar to the selected movie by comparing movie metadata.
- The poster cache file is `dataset/posters_cache.csv`.
