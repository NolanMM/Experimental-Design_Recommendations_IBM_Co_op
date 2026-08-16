# Recommendation Engine for the IBM Watson Studio Community

An end-to-end recommendation system built on real user–article interaction data from the
**IBM Watson Studio** community platform. The project implements and compares four
recommendation strategies — rank-based, user-user collaborative filtering, content-aware
lookups, and matrix factorization (SVD) — and evaluates how each behaves under sparsity
and the **cold-start problem**.

Completed as the *Experimental Design & Recommendations* capstone of the
Udacity **Data Scientist Nanodegree** (co-op portfolio project).

---

## Business Problem

IBM Watson Studio hosts thousands of articles, notebooks, and tutorials. With only
**714 of 1,051 articles** having ever been opened by a user, most content is effectively
invisible. The goal is to surface relevant articles to each user so that engagement is
spread across the catalog instead of concentrating on a handful of popular posts.

The core difficulty: the platform records **implicit feedback only** — a user either
opened an article or did not. There are no star ratings, so "relevance" has to be inferred
from co-interaction patterns rather than measured directly.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `data/user-item-interactions.csv` | 45,993 | One row per user–article interaction (`article_id`, `title`, `email`) |
| `data/articles_community.csv` | 1,056 | Article metadata (`doc_body`, `doc_description`, `doc_full_name`, `doc_status`) |

**Key statistics uncovered during EDA**

| Metric | Value |
|--------|-------|
| Unique users | 5,148 |
| Unique articles with ≥1 interaction | 714 |
| Total articles on the platform | 1,051 |
| Total interactions | 45,993 |
| Median interactions per user | 3 |
| Max interactions by a single user | 364 |
| Most-viewed article | `1429.0` — 937 views |

The distribution is heavily **long-tailed**: half of all users have interacted with three
or fewer articles, while a single power user accounts for 364 interactions. This shape is
what drives every design decision downstream.

---

## Approach

### I. Exploratory Data Analysis
- Profiled the interaction distribution per user and per article.
- Removed duplicate article records from the content table (5 duplicate `article_id`s).
- Anonymised users by mapping email hashes to sequential integer `user_id`s via a
  deterministic `email_mapper`, keeping the null-email records grouped as a single user.

### II. Rank-Based Recommendations
`get_top_articles(n)` / `get_top_article_ids(n)` return the *n* most-interacted-with
articles. Because there are no explicit ratings, **interaction count is the only available
proxy for popularity**. This is the fallback served to brand-new users, who have no history
to personalise against.

### III. User-User Collaborative Filtering
- Built a **5,149 × 714 binary user–item matrix** (1 = interacted, 0 = otherwise).
- Computed similarity as the **dot product** of user rows — for binary vectors this equals
  the count of shared articles, which is both cheap and interpretable.
- `user_user_recs_part2()` improves on a naive implementation by breaking ties
  deterministically: neighbours are ranked first by similarity, then by total interaction
  volume, and candidate articles are ranked by global popularity. This removes the
  arbitrary ordering that made the naive version non-reproducible.

### IV. Matrix Factorization (SVD)
- Applied `numpy.linalg.svd` to the user–item matrix to extract latent features.
- Unlike the classic Netflix-style setup, the matrix has **no missing values** — every
  cell is a genuine 0 or 1 — so a plain SVD is valid without FunkSVD-style imputation.
- Swept latent-feature counts from 10 → 700 and plotted reconstruction accuracy on both
  the training split (first 40,000 interactions) and the held-out test split (last 5,993).

---

## Results & Findings

**1. Cold start dominates the evaluation.** Of the 682 users in the test split, only
**20** also appear in the training set — the remaining 662 cannot be scored by SVD at all.
All 574 test articles *are* present in training, so the cold start is entirely on the
user side.

**2. More latent features ≠ better recommendations.** Training accuracy rises
monotonically with *k* — the model can memorise a matrix it has already seen. Test
accuracy moves the other way: **97.8% at k=10, falling steadily to 96.5% and flattening
past k≈390**. The extra capacity fits the training matrix without transferring to the
held-out users.

**3. Accuracy is the wrong metric here.** The scorable test subset is 20 users × 574
articles and ~98% of those cells are zeros, so a model that predicts "no interaction"
everywhere already scores ~98% — better than every SVD configuration tested. Offline
accuracy is measuring how well the model reproduces zeros, not how well it recommends.
This is the classic accuracy paradox on imbalanced implicit-feedback data. The path
forward is an **online A/B test** — rank-based control arm vs. collaborative-filtering
treatment arm — measured on click-through rate and articles-read-per-session.

**4. A layered strategy fits the data best:**

| User segment | Strategy |
|---|---|
| New user (no history) | Rank-based top-*n* |
| Light user (1–3 articles) | Content similarity + popularity blend |
| Established user | User-user collaborative filtering |
| Dense-history user | Matrix factorization with low *k* |

---

## Repository Structure

```
.
├── Recommendations_with_IBM.ipynb   # Main analysis notebook (EDA → rank → CF → SVD)
├── project_tests.py                 # Unit tests validating each stage's output
├── data/
│   ├── user-item-interactions.csv   # 45,993 user–article interactions
│   └── articles_community.csv       # 1,056 article metadata records
├── top_5.p / top_10.p / top_20.p    # Pickled expected top-article lists (test fixtures)
├── .gitignore
└── README.md
```

---

## Getting Started

**Requirements:** Python 3.10+, `pandas`, `numpy`, `matplotlib`, `jupyter`

```bash
pip install pandas numpy matplotlib jupyter
jupyter notebook Recommendations_with_IBM.ipynb
```

Run the cells top to bottom. Each section ends with an assertion or a call into
`project_tests.py`, so a clean run confirms every intermediate result — all six test
harnesses pass, and every code cell executes without error.

The notebook is self-contained: the Matrix Factorization section writes and reloads its own
`user_item_matrix.p` cache (~29 MB, gitignored) from the user–item matrix built in Part III.

> Committed outputs were produced on the original stack — Python 3.10, pandas 1.5.3,
> numpy 1.23.5, matplotlib 3.7.2. Newer pandas (3.x) has not been validated against this
> notebook.

---

## Skills Demonstrated

**Recommender Systems**
Rank-based (popularity) recommendations · User-user collaborative filtering · Content-based
filtering · Matrix factorization / Singular Value Decomposition · Latent-feature tuning ·
Cold-start mitigation · Implicit-feedback modelling

**Data Science & Analytics**
Exploratory data analysis · Long-tail distribution analysis · Data cleaning &
deduplication · Feature engineering · User anonymisation / ID mapping · Train-test
splitting on sparse matrices · Model evaluation on imbalanced data · Diagnosing the
accuracy paradox

**Experimental Design**
A/B test design for recommender evaluation · Metric selection (CTR, engagement) beyond
offline accuracy · Control-group definition · Translating offline limitations into online
experiments

**Engineering**
Python · pandas (groupby, pivot/unstack, vectorised ops) · NumPy linear algebra ·
Matplotlib · Jupyter · Test-driven analysis with assertion-based validation · Reproducible,
deterministic tie-breaking in ranking logic

---

## Acknowledgements

Data provided by **IBM Watson Studio**. Project scaffolding and test harness from the
Udacity Data Scientist Nanodegree.
