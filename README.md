﻿# 📰 NewsClassify-V2-Spacy-WordEmbeddings

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

## 🛠 Methodology

1. **Data Preprocessing**:
   - Removed stop words
   - Applied lemmatization
   - Converted text to vector representations using spaCy

2. **Model Training and Evaluation**:
   - Split data into training and testing sets
   - Trained multiple models
   - Evaluated performance using classification reports

## 🤖 Models and Results

### 1. Decision Tree Classifier

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

```
              precision    recall  f1-score   support

         0.0       0.87      0.82      0.85       579
         1.0       0.86      0.89      0.88       833
         2.0       0.88      0.88      0.88       851

    accuracy                           0.87      2263
   macro avg       0.87      0.86      0.87      2263
weighted avg       0.87      0.87      0.87      2263
```

### 5. Gradient Boosting Classifier 🏆

```
              precision    recall  f1-score   support

         0.0       0.89      0.87      0.88       579
         1.0       0.89      0.90      0.89       833
         2.0       0.90      0.90      0.90       851

    accuracy                           0.89      2263
   macro avg       0.89      0.89      0.89      2263
weighted avg       0.89      0.89      0.89      2263
```

## 🥇 Best Performing Model

The **Gradient Boosting Classifier** achieved the highest overall performance with an accuracy of 89% and balanced precision, recall, and F1-scores across all categories.

## 🚀 Future Improvements

1. Experiment with hyperparameter tuning
2. Try ensemble methods combining multiple models
3. Explore deep learning approaches (e.g., LSTM, BERT)
4. Collect more data to improve model generalization

## 📚 Dependencies

- Python 3.x
- spaCy
- scikit-learn
- NumPy
- Pandas

## 🙏 Acknowledgments

Dataset source: [Kaggle - News Category Classifier](https://www.kaggle.com/code/hengzheng/news-category-classifier-val-acc-0-65)
