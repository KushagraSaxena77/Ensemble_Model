import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_model_prediction = rf_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_model_prediction))



ada_model = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
ada_model.fit(X_train, y_train)
ada_model_prediction = ada_model.predict(X_test)

print("AdaBoostClassifier Accuracy:", accuracy_score(y_test, ada_model_prediction))



base_models = [
    ('decision_tree', DecisionTreeClassifier()),
    ('svm', SVC(probability=True))
]

meta_model = LogisticRegression()
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)
stacking_model.fit(X_train, y_train)
stacking_predictions = stacking_model.predict(X_test)

print("Stacking Model Accuracy:", accuracy_score(y_test, stacking_predictions))


