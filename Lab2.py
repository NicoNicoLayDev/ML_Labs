import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix

# =========================
# Загрузка данных
# =========================
df = pd.read_csv("students_ml_dataset.csv")

# Добавляем пропуски для реалистичности
df.loc[5, 'Attendance'] = np.nan
df.loc[12, 'StudyHours'] = np.nan

# =========================
# Разделение признаков
# =========================
X = df.drop(['FinalScore', 'DroppedOut'], axis=1)
y_reg = df['FinalScore']
y_clf = df['DroppedOut']

numeric_features = [
    'Age',
    'Attendance',
    'StudyHours',
    'AvgGrade',
    'SleepHours',
    'StressLevel'
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
# ЛИНЕЙНАЯ РЕГРЕССИЯ
# =========================
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

regression_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

regression_model.fit(X_train_reg, y_train_reg)

y_pred_reg = regression_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)

print("ЛИНЕЙНАЯ РЕГРЕССИЯ")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print()

# =========================
# ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# =========================
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=42
)

classification_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

classification_model.fit(X_train_clf, y_train_clf)

y_pred_clf = classification_model.predict(X_test_clf)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
cm = confusion_matrix(y_test_clf, y_pred_clf)

print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("Accuracy:", accuracy)
print("Confusion matrix:")
print(cm)
