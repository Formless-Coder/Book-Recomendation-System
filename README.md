# Book Recommendation System

A machine learning-based book recommendation system using K-Nearest Neighbors (KNN) algorithm to suggest similar books based on user preferences.

## 📖 Overview

This project implements a content-based book recommendation system using the **NearestNeighbors** algorithm from scikit-learn. The model recommends books similar to a given title based on features like:
- Average rating
- Number of ratings
- Language
- Rating categories (0-1, 1-2, 2-3, 3-4, 4-5)

## 🧠 Machine Learning Details

| Aspect | Details |
|--------|---------|
| **Algorithm** | K-Nearest Neighbors (KNN) |
| **Distance Metric** | Ball Tree |
| **K Value** | 6 neighbors |
| **Feature Scaling** | MinMaxScaler |
| **Features Used** | Rating categories, language codes, average rating, ratings count |

### How It Works

1. **Data Preprocessing**: 
   - Load book data from CSV
   - Create rating category bins (0-1, 1-2, 2-3, 3-4, 4-5)
   - One-hot encode language codes

2. **Feature Engineering**:
   - Combine rating categories, language dummies, average rating, and ratings count
   - Normalize features using MinMaxScaler (0-1 range)

3. **Model Training**:
   - Fit KNN model with `n_neighbors=6` and `algorithm='ball_tree'`
   - Store distances and indices for recommendation lookup

4. **Recommendation**:
   - Find the index of the input book
   - Get 6 nearest neighbors (including the book itself)
   - Return the titles of similar books

## 📁 Project Structure

```
Book-Recomendation-system/
├── book_recommender.py    # Main Python script
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── books.csv             # Dataset (not tracked in git)
└── book_recsys_venv/     # Virtual environment (not tracked)
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Formless-Coder/Book-Recomendation-System.git
cd Book-Recomendation-System
```

### 2. Create Virtual Environment
```bash
python -m venv book_recsys_venv
source book_recsys_venv/bin/activate  # Linux/Mac
# or
book_recsys_venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📦 Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

## 🔧 Usage

### Run the Python Script
```bash
python book_recommender.py
```

### Example: Get Recommendations

```python
from book_recommender import BookRecommender, df2, idlist
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load and prepare data (same as in script)
df = pd.read_csv('books.csv', on_bad_lines='skip')
# ... (preprocessing steps)

# Get recommendations
recommendations = BookRecommender('Harry Potter and the Half-Blood Prince (Harry Potter  #6)')
print(recommendations)
```

### Output
```
Recommended Books:
  - Harry Potter and the Half-Blood Prince (Harry Potter  #6)
  - Harry Potter and the Order of the Phoenix (Harry Potter  #5)
  - The Fellowship of the Ring (The Lord of the Rings  #1)
  - Harry Potter and the Chamber of Secrets (Harry Potter  #2)
  - Harry Potter and the Prisoner of Azkaban (Harry Potter  #3)
  - The Lightning Thief (Percy Jackson and the Olympians  #1)
```

## 📊 Dataset

The project uses a books dataset containing:
- **11,123 books** with 12 features including:
  - `bookID`, `title`, `authors`
  - `average_rating`, `isbn`, `isbn13`
  - `language_code`, `num_pages`, `ratings_count`
  - `text_reviews_count`, `publication_date`, `publisher`

## 🔌 Extend the Project

### Add More Features
```python
# Add publication year as a feature
df2['publication_year'] = pd.to_datetime(df2['publication_date'], errors='coerce').dt.year
features = pd.concat([rating_df, language_df, df2['average_rating'], 
                      df2['ratings_count'], df2['publication_year']], axis=1)
```

### Use Different Algorithm
```python
# Try KD Tree algorithm
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='kd_tree')
```

### Adjust K Value
```python
# Get more or fewer recommendations
model = neighbors.NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
```

## 📝 License

This project is for educational purposes.

## 👤 Author

- **GitHub**: [Formless-Coder](https://github.com/Formless-Coder)

## 🙏 Acknowledgments

- Dataset source: [Kaggle/Goodreads Books Dataset](https://www.kaggle.com)
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), and [seaborn](https://seaborn.pydata.org/)