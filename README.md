# Task 5: Text Classification on Consumer Complaint Dataset

## Overview
This project performs text classification on the **Consumer Complaint Dataset** by categorizing complaints into four classes:

| Category | Description |
|----------|------------------------------------------------|
| 0        | Credit reporting, repair, or other            |
| 1        | Debt collection                               |
| 2        | Consumer Loan                                |
| 3        | Mortgage                                     |

### Steps Involved
1. Exploratory Data Analysis (EDA) & Feature Engineering
2. Text Preprocessing
3. Data Visualization
4. Model Selection & Training
5. Model Evaluation
6. Prediction on New Data

---

## 1. Exploratory Data Analysis (EDA) & Feature Engineering
We start by loading the dataset and selecting relevant columns.

```python
import pandas as pd

# Load dataset
file_path = "/content/drive/MyDrive/consumer_complaints.csv"
df = pd.read_csv(file_path)

# Select relevant columns
df = df[['Consumer complaint narrative', 'Product']].dropna()

# Define category mapping
category_map = {'Credit reporting, repair, or other': 0, 'Debt collection': 1, 'Consumer Loan': 2, 'Mortgage': 3}
df = df[df['Product'].isin(category_map.keys())]
df['Category'] = df['Product'].map(category_map)
```

---

## 2. Text Preprocessing
We clean the complaint text using spaCy for tokenization and stopword removal.

```python
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    doc = nlp(text)
    words = [token.text for token in doc if token.text not in STOP_WORDS]
    return ' '.join(words)

# Apply text cleaning
df['Cleaned_Text'] = df['Consumer complaint narrative'].apply(clean_text)
df.head()
```

---

## 3. Data Visualization
### 1. Category Distribution
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.countplot(x=df['Category'], palette='coolwarm')
plt.xticks(ticks=[0, 1, 2, 3], labels=category_map.keys(), rotation=45)
plt.title("Category Distribution")
plt.show()
```

### 2. Word Cloud for Most Frequent Words
```python
from wordcloud import WordCloud

text = " ".join(df['Cleaned_Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Words in Complaints")
plt.show()
```

### 3. Text Length Distribution
```python
df['Text_Length'] = df['Cleaned_Text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(df['Text_Length'], bins=30, kde=True, color='blue')
plt.title("Distribution of Complaint Text Length")
plt.xlabel("Number of Words")
plt.show()
```

---

## *. Model Training
### TF-IDF Vectorization & Train-Test Split
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Text'], df['Category'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### Train Multiple Models
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"Model: {name}\nAccuracy: {accuracy_score(y_test, y_pred)}\n")
    print(classification_report(y_test, y_pred))
```

---

## 5. Model Evaluation
### Confusion Matrix for Best Model
```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

best_model = models["Logistic Regression"]
y_pred_best = best_model.predict(X_test_tfidf)

conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=category_map.keys(), yticklabels=category_map.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

## 6. Prediction on New Data
```python
def predict_complaint(text):
    cleaned_text = clean_text(text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = best_model.predict(transformed_text)[0]
    return list(category_map.keys())[list(category_map.values()).index(prediction)]

# Example Prediction
new_complaint = "I have been harassed by a debt collection agency regarding a loan I never took."
print("Predicted Category:", predict_complaint(new_complaint))
```

---

## Conclusion
- Preprocessing: Used spaCy for text cleaning.
- Visualization:Checked category distribution, word clouds, and text length.
- Model Training:Trained Naïve Bayes, Logistic Regression, and Random Forest.
- Evaluation:Used accuracy, classification report, and confusion matrix.
- Prediction:Function to classify new consumer complaints.

This implementation provides an end-to-end pipeline for text classification on consumer complaints🚀. 

# Kaiburr_Task-05
