import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

import matplotlib.pyplot as plt

# =========================
# Загрузка данных
# =========================
df = pd.read_csv("students_ml_dataset.csv")

# Добавляем пропуски
df.loc[5, 'Attendance'] = np.nan
df.loc[12, 'StudyHours'] = np.nan

# =========================
# Разделение признаков
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
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

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
# Decision Tree Classifier
# =========================
tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ))
])

tree_model.fit(X_train, y_train)

# =========================
# Предсказания
# =========================
y_pred = tree_model.predict(X_test)
y_proba = tree_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("ДЕРЕВО РЕШЕНИЙ")
print("Accuracy:", accuracy)
print("Confusion matrix:")
print(cm)

# =========================
# ROC-кривая
# =========================
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая (Decision Tree)")
plt.legend(loc="lower right")
plt.show()
