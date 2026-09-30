from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
    'title': ['title', 'movie_title', 'movie name', 'name', 'original_title'],
    'rating': ['rating', 'vote_average', 'vote average', 'imdb_rating', 'imdb rating', 'score'],
    'popularity': ['popularity', 'popularity_score', 'popularity score'],
    'votes': ['vote_count', 'vote count', 'votes', 'num_votes', 'number_of_votes'],
    'year': ['year', 'release_year', 'release year'],
    'release_date': ['release_date', 'release date', 'released', 'date'],
    'genre': ['genres', 'genre', 'genre_name', 'genre names', 'genre_ids'],
    'overview': ['overview', 'description', 'plot', 'summary'],
    'language': ['original_language', 'original language', 'language', 'lang'],
}

def normalize_name(x: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', str(x).strip().lower()).strip('_')

def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    normalized = {normalize_name(c): c for c in df.columns}
    out = {}
    for key, candidates in ALIASES.items():
        found = None
        for candidate in candidates:
            n = normalize_name(candidate)
            if n in normalized:
                found = normalized[n]
                break
        out[key] = found
    return out

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype('string').str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def add_common_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = df.copy()
    cols = detect_columns(df)
    if cols['release_date'] and cols['year'] is None:
        df[cols['release_date']] = pd.to_datetime(df[cols['release_date']], errors='coerce')
        cols['year'] = 'release_year_derived'
        df[cols['year']] = df[cols['release_date']].dt.year
    elif cols['year']:
        df[cols['year']] = pd.to_numeric(df[cols['year']], errors='coerce')
    if cols['rating']:
        df[cols['rating']] = pd.to_numeric(df[cols['rating']], errors='coerce')
    if cols['popularity']:
        df[cols['popularity']] = pd.to_numeric(df[cols['popularity']], errors='coerce')
    if cols['votes']:
        df[cols['votes']] = pd.to_numeric(df[cols['votes']], errors='coerce')
    if cols['genre']:
        df['genre_count'] = df[cols['genre']].fillna('').astype(str).apply(lambda s: len([x for x in re.split(r'[,|;/]+', s) if x.strip()]))
    if cols['year']:
        df['decade'] = (pd.to_numeric(df[cols['year']], errors='coerce') // 10 * 10).astype('Int64')
    return df, cols
