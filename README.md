# 📰 NewsClassify-V2-Spacy-WordEmbeddings

## 📊 Project Overview

This project implements a News Category Classifier using various machine learning algorithms. The goal is to automatically categorize news articles into predefined categories based on their content.

### 🔍 Dataset

The dataset consists of news articles with two main columns:
- **Text**: Description of a particular topic
- **Category**: The class to which the text belongs

Categories include:
- BUSINESS
- SPORTS
- CRIME
- SCIENCE

Distribution of categories:
```python
df['category'].value_counts()

# Output:
# BUSINESS    4254
# SPORTS      4167
# CRIME       2893
# SCIENCE     1381
# Name: count, dtype: int64
```

## 🛠 Methodology

### 1. Data Preprocessing

The preprocessing step is crucial for preparing the text data for machine learning models. Here's a detailed look at the preprocessing function:

```python
import spacy
nlp = spacy.load("en_core_web_lg")

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)
    return ' '.join(filtered_tokens)
```

This function does the following:
1. Uses spaCy to tokenize the text
2. Removes stop words (common words like "the", "is", "at", etc.)
3. Removes punctuation
4. Applies lemmatization (reducing words to their base form)

Example of preprocessed text:
```python
original_text = "Watching Schrödinger's Cat Die University of California"
preprocessed_text = "watch Schrödinger Cat Die University California"
```

### 2. Word Embeddings

Word embeddings are dense vector representations of words that capture semantic meanings. This project uses spaCy's pre-trained word vectors to create document embeddings.

```python
df['vector'] = df['preprocessed_text'].apply(lambda text: nlp(text).vector)
```

This creates a new column 'vector' that contains the vector representation of each preprocessed text. Each vector is a 300-dimensional array of floats.

Example of data structure after vector creation:
```python
print(df.head())

# Output:
#    text                                            category  label_num  preprocessed_text                    vector
# 0  Watching Schrödinger's Cat Die University of C... SCIENCE   NaN       watch Schrödinger Cat Die Univ...  [-0.85190785, 1.0438694, ...]
# 1  WATCH: Freaky Vortex Opens Up In Flooded Lake    SCIENCE   NaN       watch freaky Vortex open Flood...  [0.60747343, 1.9251899, ...]
# 2  Entrepreneurs Today Don't Need a Big Budget to... BUSINESS  2.0       entrepreneur today need Big B...   [0.088981755, 0.5882564, ...]
# ...
```

### 3. Data Preparation for Model Training

The vector data needs to be reshaped for use in scikit-learn models:

```python
import numpy as np

X_train_2d = np.stack(X_train)
X_test_2d =  np.stack(X_test)

print("Shape of X_train after reshaping: ", X_train_2d.shape)
print("Shape of X_test after reshaping: ", X_test_2d.shape)

# Output:
# Shape of X_train after reshaping:  (6789, 300)
# Shape of X_test after reshaping:  (2263, 300)
```

This reshapes the data into a 2D numpy array where each row represents a document and each column represents a dimension of the word embedding.

## 🤖 Models and Results

We experimented with several machine learning models to classify the news articles. Here are the results for each model:

### 1. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train_2d, y_train)
y_pred = clf.predict(X_test_2d)

print(classification_report(y_test, y_pred))
```

Results:
```
              precision    recall  f1-score   support

         0.0       0.71      0.65      0.68       579
         1.0       0.74      0.75      0.75       833
         2.0       0.74      0.76      0.75       851

    accuracy                           0.73      2263
   macro avg       0.73      0.72      0.73      2263
weighted avg       0.73      0.73      0.73      2263
```

### 2. Multinomial Naive Bayes

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_train_embed = scaler.fit_transform(X_train_2d)
scaled_test_embed = scaler.transform(X_test_2d)

clf = MultinomialNB()
clf.fit(scaled_train_embed, y_train)
y_pred = clf.predict(scaled_test_embed)

print(classification_report(y_test, y_pred))
```

Results:
```
              precision    recall  f1-score   support

         0.0       0.94      0.55      0.69       579
         1.0       0.72      0.85      0.78       833
         2.0       0.76      0.84      0.80       851

    accuracy                           0.77      2263
   macro avg       0.81      0.75      0.76      2263
weighted avg       0.79      0.77      0.76      2263
```

### 3. K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
clf.fit(X_train_2d, y_train)
y_pred = clf.predict(X_test_2d)

print(classification_report(y_test, y_pred))
```

Results:
```
              precision    recall  f1-score   support

         0.0       0.81      0.88      0.84       579
         1.0       0.88      0.87      0.87       833
         2.0       0.90      0.86      0.88       851

    accuracy                           0.87      2263
   macro avg       0.86      0.87      0.87      2263
weighted avg       0.87      0.87      0.87      2263
```

### 4. Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train_2d, y_train)
y_pred = clf.predict(X_test_2d)

print(classification_report(y_test, y_pred))
```

Results:
```
              precision    recall  f1-score   support

         0.0       0.87      0.82      0.85       579
         1.0       0.86      0.89      0.88       833
         2.0       0.88      0.88      0.88       851

    accuracy                           0.87      2263
   macro avg       0.87      0.86      0.87      2263
weighted avg       0.87      0.87      0.87      2263
```

### 5. Gradient Boosting Classifier

## 🥇 Best Performing Model

The **Gradient Boosting Classifier** achieved the highest overall performance with an accuracy of 89% and balanced precision, recall, and F1-scores across all categories.

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier()
clf.fit(X_train_2d, y_train)
y_pred = clf.predict(X_test_2d)

print(classification_report(y_test, y_pred))
```

Results:
```
              precision    recall  f1-score   support

         0.0       0.89      0.87      0.88       579
         1.0       0.89      0.90      0.89       833
         2.0       0.90      0.90      0.90       851

    accuracy                           0.89      2263
   macro avg       0.89      0.89      0.89      2263
weighted avg       0.89      0.89      0.89      2263
```



## 🚀 Future Improvements

1. Experiment with hyperparameter tuning
   - Use GridSearchCV or RandomizedSearchCV to find optimal parameters
2. Try ensemble methods combining multiple models
   - Voting Classifier or Stacking could potentially improve results
3. Explore deep learning approaches (e.g., LSTM, BERT)
   - These models can capture more complex relationships in text data
4. Collect more data to improve model generalization
   - More diverse examples can help the model learn better
5. Feature engineering
   - Create additional features like text length, sentiment scores, etc.

## 📚 Dependencies

- Python 3.x
- spaCy
- scikit-learn
- NumPy
- Pandas

To install dependencies:
```
pip install spacy scikit-learn numpy pandas
python -m spacy download en_core_web_lg
```

##  Acknowledgments

Dataset source: [Kaggle - News Category Classifier](https://www.kaggle.com/code/hengzheng/news-category-classifier-val-acc-0-65)

## 📖 Additional Resources

- [spaCy Documentation](https://spacy.io/usage/linguistic-features)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)
- [Introduction to Word Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)
