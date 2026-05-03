import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

df = pd.read_csv("credit_card_fraud_2025.csv")

print("Dataset shape:", df.shape)
print("\nColumns:\n", df.columns)

target_col = "Fraud_Flag"
print("\nUsing target column:", target_col)

drop_cols = ['Transaction_ID', 'Customer_ID', 'Transaction_Date']
df = df.drop(columns=drop_cols, errors='ignore')

y = df[target_col]
X = df.drop(target_col, axis=1)

print("\nClass distribution:\n", y.value_counts(normalize=True))

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred))

def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

def plot_roc(model, X_test, y_test, title):
    y_probs = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_pr(model, X_test, y_test, title):
    y_probs = model.predict_proba(X_test)[:,1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.show()

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

evaluate_model("Baseline Logistic Regression", y_test, y_pred_lr)

lr_bal = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_bal.fit(X_train_scaled, y_train)
y_pred_lr_bal = lr_bal.predict(X_test_scaled)

evaluate_model("Balanced Logistic Regression", y_test, y_pred_lr_bal)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

evaluate_model("Random Forest", y_test, y_pred_rf)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

rf_sm = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sm.fit(X_train_sm, y_train_sm)
y_pred_sm = rf_sm.predict(X_test)

evaluate_model("SMOTE + Random Forest", y_test, y_pred_sm)

plot_conf_matrix(y_test, y_pred_sm, "Confusion Matrix (SMOTE + RF)")
plot_roc(rf_sm, X_test, y_test, "ROC Curve (SMOTE + RF)")
plot_pr(rf_sm, X_test, y_test, "Precision-Recall Curve (SMOTE + RF)")

print("\n DONE")