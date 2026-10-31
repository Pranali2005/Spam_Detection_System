import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

print("--- Script Start Ho Gayi Hai ---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'spam_dataset.csv')

# 1. Dataset Check
if not os.path.exists(dataset_path):
    print(f"❌ ERROR: Mujhe '{dataset_path}' nahi mili! Pehle generate_data.py run karo.")
    exit()

print("1. Dataset mil gaya, load ho raha hai...")
df = pd.read_csv(dataset_path)

# 2. Split
print("2. Data split ho raha hai...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. Vectorize
print("3. Text ko numbers mein convert (Vectorize) kar rahe hain...")
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)

# 4. Train
print("4. AI Model train ho raha hai...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Save
print("5. Files save ho rahi hain...")
model_out_path = os.path.join(BASE_DIR, 'model.pkl')
vec_out_path = os.path.join(BASE_DIR, 'vectorizer.pkl')

with open(model_out_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(vec_out_path, 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("🎉 SUCCESS: model.pkl aur vectorizer.pkl ban gayi hain!")
print("--- Script End ---")