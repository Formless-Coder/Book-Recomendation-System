import recommender as rc
import pandas as pd

def test_recommender():
    print("Testing recommender initialization...")
    status = rc.load_and_train("books.csv")
    print(f"Status: {status}")
    
    if not status["ok"]:
        print("Initialization failed!")
        return

    titles = rc.get_all_titles()
    print(f"Total titles: {len(titles)}")
    
    test_query = "Harry Potter and the Half-Blood Prince (Harry Potter  #6)"
    print(f"\nTesting recommendation for: '{test_query}'")
    recs = rc.recommend(test_query, n=5)
    
    print(f"Found {len(recs)} recommendations:")
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['title']} by {r['authors']} (Rating: {r['average_rating']})")

    fuzzy_query = "Harry Potter"
    print(f"\nTesting fuzzy recommendation for: '{fuzzy_query}'")
    recs = rc.recommend(fuzzy_query, n=5)
    print(f"Found {len(recs)} recommendations for fuzzy query.")

if __name__ == "__main__":
    test_recommender()
