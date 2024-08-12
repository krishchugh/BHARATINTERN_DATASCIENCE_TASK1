!pip install cufflinks

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)
np.random.seed(42)
cf.set_config_file(theme='pearl')

sms_data = pd.read_csv('TASK1/SMSSpamCollection.csv', sep='\t', header=None, names=['Label', 'Message'])

print(sms_data.head())

print(f"Dataset Shape: {sms_data.shape}")

label_distribution = sms_data['Label'].value_counts()
fig = px.bar(x=label_distribution.index, y=label_distribution.values, color_discrete_sequence=[['blue', 'orange']], opacity=0.7)
iplot(fig)

spam_percentage = label_distribution['spam'] / label_distribution.sum() * 100
print(f'Spam Percentage: {spam_percentage:.2f}%')

sms_data['Label'] = sms_data['Label'].map({'ham': 0, 'spam': 1})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sms_data['Message'], sms_data['Label'], test_size=0.3, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb_classifier = MultinomialNB()
mnb_classifier.fit(X_train_vectorized, y_train)

train_predictions = mnb_classifier.predict(X_train_vectorized)
train_probs = mnb_classifier.predict_proba(X_train_vectorized)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc

train_accuracy = accuracy_score(y_train, train_predictions)
train_precision = precision_score(y_train, train_predictions)
train_recall = recall_score(y_train, train_predictions)

print(f'Training Accuracy: {train_accuracy:.3f}')
print(f'Training Precision: {train_precision:.3f}')
print(f'Training Recall: {train_recall:.3f}')

print('Training Confusion Matrix:')
print(confusion_matrix(y_train, train_predictions))

test_predictions = mnb_classifier.predict(X_test_vectorized)

test_conf_matrix = confusion_matrix(y_test, test_predictions)
TN, FP, FN, TP = test_conf_matrix.ravel()

test_sensitivity = TP / (TP + FN)
test_specificity = TN / (TN + FP)
test_precision = TP / (TP + FP)
test_recall = TP / (TP + FN)

print(f'Test Sensitivity: {test_sensitivity:.3f}')
print(f'Test Specificity: {test_specificity:.3f}')
print(f'Test Precision: {test_precision:.3f}')
print(f'Test Recall: {test_recall:.3f}')
print(f'Test F1 Score: {f1_score(y_test, test_predictions):.3f}')

test_probs = mnb_classifier.predict_proba(X_test_vectorized)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, test_probs)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print(f'ROC AUC: {roc_auc_value:.3f}')

from sklearn.naive_bayes import BernoulliNB
bnb_classifier = BernoulliNB()
bnb_classifier.fit(X_train_vectorized, y_train)

bnb_test_predictions = bnb_classifier.predict(X_test_vectorized)
bnb_test_probs = bnb_classifier.predict_proba(X_test_vectorized)[:, 1]

bnb_conf_matrix = confusion_matrix(y_test, bnb_test_predictions)
bnb_TN, bnb_FP, bnb_FN, bnb_TP = bnb_conf_matrix.ravel()

bnb_sensitivity = bnb_TP / (bnb_TP + bnb_FN)
bnb_specificity = bnb_TN / (bnb_TN + bnb_FP)
bnb_precision = bnb_TP / (bnb_TP + bnb_FP)
bnb_recall = bnb_TP / (bnb_TP + bnb_FN)

print(f'BernoulliNB Sensitivity: {bnb_sensitivity:.3f}')
print(f'BernoulliNB Specificity: {bnb_specificity:.3f}')
print(f'BernoulliNB Precision: {bnb_precision:.3f}')
print(f'BernoulliNB Recall: {bnb_recall:.3f}')
print(f'BernoulliNB F1 Score: {f1_score(y_test, bnb_test_predictions):.3f}')

bnb_fpr, bnb_tpr, bnb_thresholds = roc_curve(y_test, bnb_test_probs)
bnb_roc_auc_value = auc(bnb_fpr, bnb_tpr)

plt.figure(figsize=(8, 6))
plt.plot(bnb_fpr, bnb_tpr, color='blue', lw=2)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('BernoulliNB ROC Curve')
plt.show()
print(f'BernoulliNB ROC AUC: {bnb_roc_auc_value:.3f}')


