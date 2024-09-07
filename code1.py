import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# dataset
df = pd.read_csv('disaster_tweets_data(DS)(1).csv')

# null values
print("Null values in each column before handling:\n", df.isnull().sum())

# Handle null values
df = df.dropna()  # Remove rows

print("\nNull values in each column after handling:\n", df.isnull().sum())

def preprocess_text(text):
    # lowercase
    text = text.lower()
    # punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    return text

# preprocessing
df['tweets'] = df['tweets'].apply(preprocess_text)

# tweets and target variables
X = df['tweets']
y = df['target']

# training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF with stopwords removal
vectorizer = TfidfVectorizer(stop_words='english')  # Using built-in stopwords
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Multinomial Naïve Bayes Classification
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

# KNN Classification
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_vec, y_train)
y_pred_knn = knn_model.predict(X_test_vec)

# Confusion Matrix and Classification Report
nb_confusion_matrix = confusion_matrix(y_test, y_pred_nb)
nb_classification_report = classification_report(y_test, y_pred_nb)

lr_confusion_matrix = confusion_matrix(y_test, y_pred_lr)
lr_classification_report = classification_report(y_test, y_pred_lr)

knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_classification_report = classification_report(y_test, y_pred_knn)

# best model based on accuracy
nb_accuracy = nb_model.score(X_test_vec, y_test)
lr_accuracy = lr_model.score(X_test_vec, y_test)
knn_accuracy = knn_model.score(X_test_vec, y_test)

best_model = max(nb_accuracy, lr_accuracy, knn_accuracy)
best_model_name = (
    "Multinomial Naive Bayes" if best_model == nb_accuracy else 
    "Logistic Regression" if best_model == lr_accuracy else 
    "KNN Classification"
)

# results
print("Multinomial Naive Bayes Confusion Matrix:\n", nb_confusion_matrix)
print("\nMultinomial Naive Bayes Classification Report:\n", nb_classification_report)

print("\nLogistic Regression Confusion Matrix:\n", lr_confusion_matrix)
print("\nLogistic Regression Classification Report:\n", lr_classification_report)

print("\nKNN Classification Confusion Matrix:\n", knn_confusion_matrix)
print("\nKNN Classification Classification Report:\n", knn_classification_report)

best_accuracy = max(nb_accuracy, lr_accuracy, knn_accuracy)
if best_accuracy == nb_accuracy:
    best_model_name = "Multinomial Naive Bayes"
elif best_accuracy == lr_accuracy:
    best_model_name = "Logistic Regression"
else:
    best_model_name = "KNN Classification"

# Output the results
print("\nMultinomial Naive Bayes Accuracy: {:.4f}".format(nb_accuracy))
print("Logistic Regression Accuracy: {:.4f}".format(lr_accuracy))
print("KNN Classification Accuracy: {:.4f}".format(knn_accuracy))

print("\nBest Model Based on Accuracy: ", best_model_name)
print("Best Model Accuracy: {:.4f}".format(best_accuracy))