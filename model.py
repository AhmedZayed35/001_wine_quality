import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix



df = pd.read_csv('./data/winequality.csv')

# cleaning and visualising
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col].fillna(df[col].mean(), inplace=True)

# df histogram     
# df.hist(bins=20, figsize=(10,10))
# plt.show()

# wine quality to alchol plot
# plt.bar(df.quality, df.alcohol)
# plt.xlabel('quality')
# plt.ylabel('alcohol')
# plt.savefig("assets/wine_quality_to_alchol_hist.png", dpi=300)
# plt.show()


# heat map
plt.figure(figsize=(12, 12))
sb.heatmap(df.iloc[:, 1:].corr() > 0.7, annot=True, cbar=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap (Threshold > 0.7)", fontsize=14)
plt.savefig('assets/heatmap_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# model
df = df.drop('total sulfur dioxide', axis=1)
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)

features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=40)
print(xtrain.shape, xtest.shape)
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.fit_transform(xtest)

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
model_names = ['Logistic Regression', 'XGBoost Classifier', 'Support Vector Classifier']

for i in range(3):
    models[i].fit(xtrain, ytrain)
    
    print(f'{models[i]}: ')
    print('Trainig Accuracy: ' , metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))


conf_matrices = [confusion_matrix(ytest, model.predict(xtest)) for model in models]

# Plot combined confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices for Wine Quality Classification Models', fontsize=16, fontweight='bold')

for idx, ax in enumerate(axes):
    sb.heatmap(conf_matrices[idx], annot=True, fmt='d', cmap="Blues", ax=ax, cbar=False,
               annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(model_names[idx], fontsize=14, pad=10)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('assets/confusion_matrices.png', dpi=300)
plt.show()


for i in range(3):
    print(f'{models[i]}: ')
    print('Trainig Accuracy: ' , metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
    print('Validation Accuracy : ', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
    print('Classification Report: ')
    print(metrics.classification_report(ytrain, models[i].predict(xtrain)))
print(metrics.classification_report(ytest, models[2].predict(xtest)))
