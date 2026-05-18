const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const dropdown = document.getElementById("dropdown");
const trendingGrid = document.getElementById("trendingGrid");

let debounceTimer = null;

// Get first letter of movie title for gradient poster placeholder
function getInitials(title) {
  return title ? title[0].toUpperCase() : "?";
}

// Generate a deterministic gradient pair from movie title (hash → color)
function getGradient(title) {
  const colors = [
    ["#e50914", "#6c63ff"],
    ["#ff6b35", "#f7c59f"],
    ["#00b4d8", "#0077b6"],
    ["#06d6a0", "#118ab2"],
    ["#7209b7", "#3a0ca3"],
    ["#f72585", "#b5179e"],
    ["#e63946", "#457b9d"],
    ["#2ec4b6", "#ff9f1c"],
  ];
  let hash = 0;
  for (let i = 0; i < title.length; i++) {
    hash = title.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
}

// Build poster HTML — real image if cached, otherwise gradient + initial letter
function posterHTML(movie, size) {
  const [c1, c2] = getGradient(movie.title);
  const fontSize = size === "small" ? "3.5rem" : "5rem";
  if (movie.poster_url) {
    return `<div class="card-poster" style="background: linear-gradient(135deg, ${c1}, ${c2})">
      <img class="poster-img" src="${movie.poster_url}" alt="${movie.title.replace(/'/g, "\\'")}" loading="lazy">
      <span class="rating-badge">${movie.vote_average || "—"}</span>
      ${movie.similarity ? `<span class="match-badge">${movie.similarity}% Match</span>` : ""}
    </div>`;
  }
  return `<div class="card-poster" style="background: linear-gradient(135deg, ${c1}, ${c2})">
    <span class="poster-initial" style="font-size:${fontSize}">${getInitials(movie.title)}</span>
    <span class="rating-badge">${movie.vote_average || "—"}</span>
    ${movie.similarity ? `<span class="match-badge">${movie.similarity}% Match</span>` : ""}
  </div>`;
}

// Build full-size movie card HTML (for trending grid)
function renderCard(movie) {
  const genres = Array.isArray(movie.genres) ? movie.genres.join(", ") : movie.genres || "";
  const overview = movie.overview
    ? (movie.overview.length > 120 ? movie.overview.slice(0, 120) + "..." : movie.overview)
    : "";
  return `
    <div class="movie-card" data-title="${movie.title.replace(/'/g, "\\'")}">
      ${posterHTML(movie, "large")}
      <div class="card-body">
        <h3>${movie.title}</h3>
        <p class="genres">${genres}</p>
        <p class="overview">${overview}</p>
      </div>
    </div>
  `;
}

// Build small recommendation card HTML (for horizontal scroll strip)
function renderRecCard(movie) {
  const genres = Array.isArray(movie.genres) ? movie.genres.slice(0, 2).join(", ") : movie.genres || "";
  return `
    <div class="recs-card" data-title="${movie.title.replace(/'/g, "\\'")}">
      ${posterHTML(movie, "small")}
      <div class="card-body">
        <h3>${movie.title}</h3>
        <p class="genres">${genres}</p>
      </div>
    </div>
  `;
}

// Click a movie card → fetch recommendations → show inline strip below it
function fetchAndShowRecs(title, cardElement) {
  const grid = cardElement.closest(".movie-grid");
  if (!grid) return;

  const existing = document.querySelector(".recs-container");
  if (existing) existing.remove();

  const container = document.createElement("div");
  container.className = "recs-container";
  container.innerHTML = `
    <div class="recs-label">
      <span class="accent-bar"></span>
      More Like <span>${title}</span>
    </div>
    <div class="recs-row" id="recsRow">
      <div class="recs-spinner"></div>
    </div>
  `;
  cardElement.after(container);
  container.scrollIntoView({ behavior: "smooth", block: "nearest" });

  fetch("/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  })
    .then((r) => r.json())
    .then((data) => {
      const row = document.getElementById("recsRow");
      if (data.recommendations && data.recommendations.length > 0) {
        row.innerHTML = data.recommendations.map(renderRecCard).join("");
        row.querySelectorAll(".recs-card").forEach((recCard) => {
          recCard.addEventListener("click", (e) => {
            e.stopPropagation();
            const recTitle = recCard.dataset.title;
            const existingStrip = recCard.closest(".recs-container");
            if (existingStrip) existingStrip.remove();
            if (grid) {
              const newContainer = document.createElement("div");
              newContainer.className = "recs-container";
              newContainer.innerHTML = `
                <div class="recs-label">
                  <span class="accent-bar"></span>
                  More Like <span>${recTitle}</span>
                </div>
                <div class="recs-row" id="recsRowCascade">
                  <div class="recs-spinner"></div>
                </div>
              `;
              const parentRecs = recCard.closest(".recs-container");
              if (parentRecs) {
                parentRecs.after(newContainer);
                newContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
                fetch("/recommend", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ title: recTitle }),
                })
                  .then((r) => r.json())
                  .then((newData) => {
                    const cascadeRow = document.getElementById("recsRowCascade");
                    if (newData.recommendations && newData.recommendations.length > 0) {
                      cascadeRow.innerHTML = newData.recommendations.map(renderRecCard).join("");
                      cascadeRow.querySelectorAll(".recs-card").forEach((cRecCard) => {
                        cRecCard.addEventListener("click", (ce) => {
                          ce.stopPropagation();
                          const cRecTitle = cRecCard.dataset.title;
                          const cs = cRecCard.closest(".recs-container");
                          if (cs) cs.remove();
                          fetchAndShowRecs(cRecTitle, cRecCard);
                        });
                      });
                    } else {
                      cascadeRow.innerHTML = '<p style="color:#888;padding:20px;">No recommendations found.</p>';
                    }
                  });
              }
            }
          });
        });
      } else {
        row.innerHTML = '<p style="color:#888;padding:20px;">No recommendations found for this movie.</p>';
      }
    })
    .catch(() => {
      const row = document.getElementById("recsRow");
      row.innerHTML = '<p style="color:#888;padding:20px;">Something went wrong. Try again.</p>';
    });
}

// Attach click handler to every movie card
function attachCardClick(card) {
  card.addEventListener("click", () => {
    const title = card.dataset.title;
    if (title) {
      fetchAndShowRecs(title, card);
    }
  });
}

// Apply click handler to all trending cards on page load
document.querySelectorAll(".movie-card").forEach(attachCardClick);

searchInput.addEventListener("input", function () {
  clearTimeout(debounceTimer);
  const q = this.value.trim();
  if (q.length < 2) {
    dropdown.classList.remove("active");
    return;
  }
  debounceTimer = setTimeout(() => {
    fetch("/api/search?q=" + encodeURIComponent(q))
      .then((r) => r.json())
      .then((titles) => {
        dropdown.innerHTML = titles
          .map((t) => `<div class="dropdown-item" data-title="${t.replace(/'/g, "\\'")}">${t}</div>`)
          .join("");
        dropdown.classList.add("active");
        dropdown.querySelectorAll(".dropdown-item").forEach((item) => {
          item.addEventListener("click", () => {
            searchInput.value = item.dataset.title;
            dropdown.classList.remove("active");
            const targetCard = document.querySelector(`.movie-card[data-title="${item.dataset.title.replace(/'/g, "\\'")}"]`);
            if (targetCard) {
              fetchAndShowRecs(item.dataset.title, targetCard);
            } else {
              const virtualCard = document.createElement("div");
              virtualCard.dataset.title = item.dataset.title;
              virtualCard.style.display = "none";
              trendingGrid.appendChild(virtualCard);
              fetchAndShowRecs(item.dataset.title, virtualCard);
            }
          });
        });
      });
  }, 250);
});

// Search button click — trigger recommendations for typed title
searchBtn.addEventListener("click", () => {
  const title = searchInput.value.trim();
  if (title) {
    const targetCard = document.querySelector(`.movie-card[data-title="${title.replace(/'/g, "\\'")}"]`);
    if (targetCard) {
      fetchAndShowRecs(title, targetCard);
    } else {
      const virtualCard = document.createElement("div");
      virtualCard.dataset.title = title;
      virtualCard.style.display = "none";
      trendingGrid.appendChild(virtualCard);
      fetchAndShowRecs(title, virtualCard);
    }
  }
});

// Enter key in search field — same as clicking search button
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    const title = searchInput.value.trim();
    if (title) {
      dropdown.classList.remove("active");
      const targetCard = document.querySelector(`.movie-card[data-title="${title.replace(/'/g, "\\'")}"]`);
      if (targetCard) {
        fetchAndShowRecs(title, targetCard);
      } else {
        const virtualCard = document.createElement("div");
        virtualCard.dataset.title = title;
        virtualCard.style.display = "none";
        trendingGrid.appendChild(virtualCard);
        fetchAndShowRecs(title, virtualCard);
      }
    }
  }
});

// Close autocomplete dropdown when clicking outside the search box
document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-box")) {
    dropdown.classList.remove("active");
  }
});
