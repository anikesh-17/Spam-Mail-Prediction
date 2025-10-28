# Spam Mail Detection using Logistic Regression

This project demonstrates how to build a **Spam Mail Detection System**
using **Machine Learning (Logistic Regression)**. It uses the **TF-IDF
(Term Frequency--Inverse Document Frequency)** approach to convert text
messages into numerical feature vectors.

------------------------------------------------------------------------

## 📘 Project Overview

Spam detection is a classic Natural Language Processing (NLP) problem.
The goal is to classify a given message as either:
- **Spam (0)** --Unwanted promotional or fraudulent messages
- **Ham (1)** -- Legitimate messages

The dataset used here is the `mail_data.csv` file which contains labeled
SMS messages.

------------------------------------------------------------------------

## 🧩 Steps and Concepts Explained

### 1. Importing Dependencies

We import all the required Python libraries:

``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

-   **pandas, numpy:** For data handling and preprocessing
-   **train_test_split:** To divide the dataset into training and
    testing sets
-   **TfidfVectorizer:** Converts text into numerical form
-   **LogisticRegression:** For building the classification model
-   **accuracy_score:** To evaluate model performance

------------------------------------------------------------------------

### 2. Data Collection and Preprocessing

The dataset is read using pandas and null values are replaced with empty
strings.

``` python
raw_mail_data = pd.read_csv('/content/mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
```

We inspect data shape and head to understand structure:

``` python
mail_data.shape  # (5572, 2)
mail_data.head()
```

------------------------------------------------------------------------

### 3. Label Encoding

The `Category` column contains **'ham'** and **'spam'** labels.\
We encode them into numeric values:

``` python
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
```

-   **Spam = 0**
-   **Ham = 1**

Then, separate input features and output labels:

``` python
X = mail_data['Message']
Y = mail_data['Category']
```

------------------------------------------------------------------------

### 4. Splitting the Data

The dataset is divided into training and testing sets using 80--20
split.

``` python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
```

------------------------------------------------------------------------

### 5. Feature Extraction (TF-IDF)

We use **TfidfVectorizer** to convert text into numeric feature vectors.

``` python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
```

-   **TF-IDF** gives importance to words based on how frequently they
    appear across documents.

Convert labels to integer format:

``` python
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')
```

------------------------------------------------------------------------

### 6. Model Training (Logistic Regression)

Train a Logistic Regression model on the extracted features:

``` python
model = LogisticRegression()
model.fit(X_train_features, Y_train)
```

------------------------------------------------------------------------

### 7. Model Evaluation

We evaluate model performance using **accuracy score**:

``` python
# Training accuracy
train_prediction = model.predict(X_train_features)
train_accuracy = accuracy_score(Y_train, train_prediction)

# Test accuracy
test_prediction = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_prediction)
```

Example output:

    Accuracy on training data :  0.9676
    Accuracy on test data :  0.9668

The model performs very well, indicating effective generalization.

------------------------------------------------------------------------

### 8. Building a Predictive System

To test the model with a new message:

``` python
input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575."]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)
```

Output:

    [0]
    spam mail

------------------------------------------------------------------------

## 🚀 Key Takeaways

-   TF-IDF helps transform textual data into machine-understandable
    numeric format.\
-   Logistic Regression is a simple yet powerful algorithm for binary
    classification.\
-   The model achieves **over 96% accuracy** on both training and
    testing datasets.\
-   You can easily extend this to real-time email or SMS filtering
    systems.

------------------------------------------------------------------------

## 📂 Technologies Used

-   **Python 3**
-   **Pandas, NumPy**
-   **scikit-learn (sklearn)**

------------------------------------------------------------------------

## 🧠 Author

**Anikesh Sharma**\
B.Tech in Computer Science \| Machine Learning Enthusiast

------------------------------------------------------------------------
