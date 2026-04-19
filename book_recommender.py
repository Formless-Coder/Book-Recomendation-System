"""
Book Recommendation System
A KNN-based book recommendation model using NearestNeighbors algorithm.
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# DATA LOADING
# ============================================================
df = pd.read_csv('books.csv', on_bad_lines='skip')
print("Data loaded successfully!")
print(f"Shape: {df.shape}")
df.head()


# ============================================================
# DATA EXPLORATION
# ============================================================

# Check for null values
print("\nNull values:")
print(df.isnull().sum())

# Statistical summary
print("\nData description:")
print(df.describe())

# Top 10 books by rating (with > 1M ratings)
top_ten = df[df['ratings_count'] > 1000000]
top_ten.sort_values(by='average_rating', ascending=False)
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 10))
data = top_ten.sort_values(by='average_rating', ascending=False).head(10)
sns.barplot(x="average_rating", y="title", data=data, palette='inferno')
plt.show()

# Top 10 authors with most books
most_books = df.groupby('authors')['title'].count().reset_index().sort_values('title', ascending=False).head(10).set_index('authors')
plt.figure(figsize=(15,10))
ax = sns.barplot(x='title', y=most_books.index, data=most_books, palette='inferno')
ax.set_title("Top 10 authors with most books")
ax.set_xlabel("Total number of books")
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2, str(round(i.get_width())), fontsize=15, color='black')
plt.show()

# Most rated books
most_rated = df.sort_values('ratings_count', ascending=False).head(10).set_index('title')
plt.figure(figsize=(15,10))
ax = sns.barplot(x='ratings_count', y=most_rated.index, data=most_rated, palette='inferno')
for i in ax.patches:
    ax.text(i.get_width()+.2, i.get_y()+.2, str(round(i.get_width())), fontsize=15, color='black')
plt.show()

# Average rating distribution
df.average_rating = df.average_rating.astype(float)
fig, ax = plt.subplots(figsize=[15,10])
sns.histplot(df['average_rating'], ax=ax)
ax.set_title('Average rating distribution for all books', fontsize=20)
ax.set_xlabel('Average rating', fontsize=13)
plt.show()

# Relation between Rating counts and Average Ratings
ax = sns.relplot(data=df, x="average_rating", y="ratings_count", color='red', sizes=(100, 200), height=7, marker='o')
plt.title("Relation between Rating counts and Average Ratings", fontsize=15)
ax.set_axis_labels("Average Rating", "Ratings Count")
plt.show()

# Relation between Average Rating and Number of Pages
plt.figure(figsize=(15,10))
ax = sns.relplot(x="average_rating", y="  num_pages", data=df, color='red', sizes=(100, 200), height=7, marker='o')
ax.set_axis_labels("Average Rating", "Number of Pages")
plt.show()


# ============================================================
# DATA PREPARATION
# ============================================================
df2 = df.copy()

# Create rating categories
df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

# One-hot encoding
rating_df = pd.get_dummies(df2['rating_between'])
language_df = pd.get_dummies(df2['language_code'])

# Create feature matrix
features = pd.concat([
    rating_df, 
    language_df, 
    df2['average_rating'], 
    df2['ratings_count']
], axis=1)


# ============================================================
# MODEL TRAINING
# ============================================================
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# KNN model using Ball Tree algorithm
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================
def BookRecommender(book_name):
    """
    Get book recommendations based on a given book title.
    
    Args:
        book_name: Title of the book to find recommendations for
        
    Returns:
        List of recommended book titles
    """
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df2.loc[newid].title)
    return book_list_name


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    BookNames = BookRecommender('Harry Potter and the Half-Blood Prince (Harry Potter  #6)')
    print("Recommended Books:")
    for book in BookNames:
        print(f"  - {book}")