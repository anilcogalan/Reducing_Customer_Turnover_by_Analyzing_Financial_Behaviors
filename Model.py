import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Preprocessing_Featues import df
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, classification_report
from Preprocessing_Featues import X_train,y_train,X_test,y_test
from sklearn.model_selection import cross_val_score
from pycaret.datasets import get_data
from pycaret.classification import setup, compare_models, finalize_model, predict_model

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x)
pd.set_option('display.width', 500)




##################################
# WHICH MODEL IS THE BEST ?
##################################

# Set up a classification experiment using PyCaret
exp = setup(df, target='e_signed')

# Compare the performance of different machine learning models
best_model = compare_models()

final_model = finalize_model(best_model)
predictions = predict_model(final_model, df)

# Light Gradient Boosting Machine

##################################
# Light Gradient Boosting Machine
##################################

classifier = lgb.LGBMClassifier(random_state = 0).fit(X_train, y_train)

cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
print(cv_scores)

# Predicting Test Set
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
auc = roc_auc_score(y_test, classifier.predict_proba(X_test)[:,1])


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(auc, 2)}")
print("Classification report:\n", report)


# Accuracy: 0.65
# Recall: 0.661
# Precision: 0.71
# F1: 0.68
# Auc: 0.71

