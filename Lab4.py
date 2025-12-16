import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

import matplotlib.pyplot as plt

# =========================
# Загрузка данных
# =========================
df = pd.read_csv("students_ml_dataset.csv")

# Добавляем пропуски
df.loc[5, 'Attendance'] = np.nan
df.loc[12, 'StudyHours'] = np.nan

# =========================
# Признаки и цель
# =========================
X = df.drop(['DroppedOut'], axis=1)
y = df['DroppedOut']

numeric_features = [
    'Age',
    'Attendance',
    'StudyHours',
    'AvgGrade',
    'SleepHours',
    'StressLevel',
    'FinalScore'
]

categorical_features = [
    'PartTimeJob',
    'Faculty'
]

# =========================
# Препроцессор
# =========================
numeric_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =========================
# Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# СЛУЧАЙНЫЙ ЛЕС (OOB)
# =========================
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        oob_score=True,
        bootstrap=True
    ))
])

rf_model.fit(X_train, y_train)

rf_clf = rf_model.named_steps['model']

print("RANDOM FOREST")
print("OOB Accuracy:", rf_clf.oob_score_)
print("OOB Error:", 1 - rf_clf.oob_score_)
print()

# =========================
# AdaBoost
# =========================
ada_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', AdaBoostClassifier(
        n_estimators=200,
        random_state=42
    ))
])

ada_model.fit(X_train, y_train)

ada_proba = ada_model.predict_proba(X_test)[:, 1]

# =========================
# Gradient Boosting
# =========================
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    ))
])

gb_model.fit(X_train, y_train)

gb_proba = gb_model.predict_proba(X_test)[:, 1]

# =========================
# ROC-кривые
# =========================
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_proba)
roc_auc_ada = auc(fpr_ada, tpr_ada)

fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_proba)
roc_auc_gb = auc(fpr_gb, tpr_gb)

plt.figure()
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost (AUC = {roc_auc_ada:.2f})")
plt.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC = {roc_auc_gb:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые (Ансамблевые методы)")
plt.legend()
plt.show()
