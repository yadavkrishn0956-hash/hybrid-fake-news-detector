import numpy as np
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split ,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score,
precision_score,
recall_score, 
f1_score,
confusion_matrix,
classification_report)

#Load dataset
print("Loading datasets..")
try:
    fake = pd.read_csv("data/Fake.csv")
    real = pd.read_csv("data/True.csv")
    fake_ = fake.drop_duplicates().copy()
    fake_["label"] = 0
    real_ = real.drop_duplicates().copy()
    real_["label"] = 1
    dataset = pd.concat([real_,fake_])
    dataset["content"] = dataset["title"] + " " + dataset["text"]
except FileNotFoundError:
    print("CSV files not found in /data folder.")
    exit() 

#train/test-split
x = dataset['content']
y = dataset['label']
x_train , x_test , y_train , y_test = train_test_split(x, y, test_size = 0.2, random_state=42,stratify=y)

#Vectorization & Training
vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

#Save Artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Saved model.pkl and vectorizer.pkl successfully!")
