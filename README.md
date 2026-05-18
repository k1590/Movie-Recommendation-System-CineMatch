# CineMatch – AI Powered Movie Recommendation System

## Project Overview

CineMatch is an AI-powered Movie Recommendation System developed using Machine Learning techniques. The system recommends movies similar to a user-selected movie using a Content-Based Filtering approach.

The project analyzes movie metadata such as genres, keywords, and movie overview to identify similarities between movies and generate recommendations efficiently.

---

# Features

- AI-powered movie recommendations  
- Content-Based Filtering using Machine Learning  
- Fast recommendation generation  
- Interactive and responsive user interface  
- Movie search with autocomplete  
- Dynamic movie detail pages  
- Trending movie section  
- Movie posters and ratings display  
- Flask backend integration  

---

# Machine Learning Concepts Used

- Content-Based Filtering
- CountVectorizer
- Cosine Similarity
- Text Preprocessing
- Feature Extraction

---

# Technologies Used

## Frontend
- HTML
- CSS
- JavaScript

## Backend
- Python
- Flask

## Machine Learning
- Scikit-learn
- Pandas
- NumPy

---

# Dataset Used

Dataset Name: TMDB 5000 Movie Dataset  
Source: Kaggle

Dataset contains:
- 4,803 movies
- Genres
- Keywords
- Overview
- Ratings
- Popularity

---

# System Workflow

1. Load TMDB movie dataset  
2. Preprocess genres, keywords, and overview  
3. Create combined tags column  
4. Convert text into vectors using CountVectorizer  
5. Calculate movie similarity using Cosine Similarity  
6. Generate top 10 similar movie recommendations  
7. Display recommendations dynamically on webpage  

---

# Recommendation Algorithm

## CountVectorizer
Converts textual movie data into numerical vectors.

## Cosine Similarity

Formula:

cos(θ) = (A · B) / (||A|| × ||B||)

Used to measure similarity between movie vectors.

---

# Project Structure

Movie-Recommendation-System/

├── app.py  
├── recommendation.py  
├── templates/  
├── static/  
├── dataset/  
├── requirements.txt  
└── README.md  

---

# Installation and Setup

## Clone Repository

```bash
git clone <repository-url>
cd Movie-Recommendation-System
```
