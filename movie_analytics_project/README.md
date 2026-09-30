# Movie Ratings, Genres & Popularity — End-to-End Data Science Project

Source dataset: Kaggle — https://www.kaggle.com/datasets/rishabhchaudhary07/top-movies-ratings-genres-popularity-and-metadata

## Project goal
Analyze movie ratings, popularity, genres, release patterns and available metadata; then build reusable ML components and a Streamlit dashboard.

## Workflow
1. Data ingestion
2. Data audit
3. Cleaning and validation
4. Univariate / bivariate / multivariate EDA
5. Feature engineering
6. Statistical relationships
7. ML: rating regression + high-rating classification
8. Optional content-based recommendation
9. Streamlit dashboard
10. Business-style insights and documentation

## Setup
```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

Put the Kaggle CSV inside `data/` and rename it to `movies.csv`.

Then run:
```bash
jupyter notebook
```
Open the notebooks in order.

For dashboard:
```bash
streamlit run app.py
```

## Important
The notebook is deliberately defensive: it detects common variants of title/rating/popularity/year/genre/overview columns instead of assuming one fixed schema.
