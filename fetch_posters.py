import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_KEY = "32f6e90594fba04c6ec8b52623adbe4e"
BASE_URL = "https://api.themoviedb.org/3/movie"
CACHE_PATH = "dataset/posters_cache.csv"
MOVIES_PATH = "dataset/tmdb_5000_movies.csv"

df = pd.read_csv(MOVIES_PATH)
ids = df["id"].tolist()
total = len(ids)

try:
    cached = pd.read_csv(CACHE_PATH)
    cached_ids = set(cached["id"].tolist())
    print(f"Resuming — {len(cached)} already cached")
except FileNotFoundError:
    cached_ids = set()

session = requests.Session()

def fetch_poster(mid):
    if mid in cached_ids:
        return None
    try:
        resp = session.get(f"{BASE_URL}/{mid}", params={"api_key": API_KEY}, timeout=10)
        if resp.status_code == 200:
            poster = resp.json().get("poster_path")
            if poster:
                return {"id": mid, "poster_path": poster}
        return None
    except:
        return None

pending = [mid for mid in ids if mid not in cached_ids]
print(f"Need to fetch: {len(pending)} movies")

batch_size = 40
for i in range(0, len(pending), batch_size):
    batch = pending[i : i + batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(pending) - 1) // batch_size + 1
    print(f"Batch {batch_num}/{total_batches}...", end=" ")

    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        fut = {ex.submit(fetch_poster, mid): mid for mid in batch}
        for f in as_completed(fut):
            r = f.result()
            if r:
                results.append(r)

    if results:
        pd.DataFrame(results).to_csv(
            CACHE_PATH, mode="a", index=False,
            header=not bool(pd.io.common.file_exists(CACHE_PATH)),
        )

    found = len(results)
    missing = len(batch) - found
    print(f"{found} found, {missing} missing")

    if i + batch_size < len(pending):
        time.sleep(10)

final = pd.read_csv(CACHE_PATH)
print(f"\nDone! Total: {total}")
print(f"Cached posters: {len(final)}")
print(f"Missing: {total - len(final)}")
