# 📚 Premium Book Recommendation System

A modern, high-performance web application that delivers personalized book recommendations using a **K-Nearest Neighbors (KNN)** machine learning engine. Built with a premium Gradio interface and optimized for speed and accuracy.

---

## 🌟 Key Features

- **🚀 Lightning Fast Engine**: Core recommendation lookup completes in less than 0.01 seconds.
- **🎨 Premium UI**: A stunning "Midnight Slate" interface featuring glassmorphism, smooth animations, and high-contrast typography.
- **🔍 Intelligent Search**: Lightweight search input with intelligent fuzzy matching and substring fallback.
- **🛠️ Modular Architecture**: Clean separation between the ML engine (`recommender.py`) and the web interface (`app.py`).
- **📊 Robust Analysis**: Uses the Kaggle Goodreads dataset (11,000+ books) with feature engineering on ratings, language, and popularity.

---

## ⚙️ Tech Stack

- **Machine Learning**: `scikit-learn` (Nearest Neighbors with Ball Tree algorithm).
- **Data Processing**: `pandas`, `numpy`.
- **Frontend**: `Gradio` 6.x (Custom CSS & Theme).
- **Backend**: Python 3.9+.

---

## 📁 Project Structure

```text
Book-Recomendation-System/
├── app.py               # Premium Gradio Web Interface
├── recommender.py       # Core ML Engine & Logic
├── books.csv            # Dataset (11,000+ books) - [Download from Kaggle]
├── requirements.txt      # Updated dependencies
├── test_recommender.py  # Diagnostic & Sanity check script
└── .gitignore           # Version control rules
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Formless-Coder/Book-Recomendation-System.git
cd Book-Recomendation-System
pip install -r requirements.txt
```

### 2. Dataset Setup
Ensure `books.csv` is present in the root directory. You can download it from the [Goodreads Books Dataset on Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks).

### 3. Running the App
Start the Gradio web server:

```bash
export PYTHONPATH=$PYTHONPATH:.
python app.py
```
Then open the local URL provided (usually `http://127.0.0.1:7860`) in your browser.

---

## 🧠 Recommendation Logic

The system uses a content-based filtering approach with the following steps:
1. **Feature Engineering**: Books are mapped to a high-dimensional space based on their average rating, ratings count, and language.
2. **One-Hot Encoding**: Language codes and rating buckets are transformed into numeric features.
3. **MinMax Scaling**: All features are normalized to a 0-1 range to ensure fair distance calculations.
4. **KNN Search**: The **Ball Tree** algorithm is used to find the 5 closest neighbors (most similar books) in the feature space.

---

## ⚖️ License
This project is for educational purposes. All book data belongs to their respective publishers and authors.

Built with ❤️ by Formless-Coder.
