import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("students_ml_dataset.csv")

df.loc[5, 'Attendance'] = np.nan
df.loc[12, 'StudyHours'] = np.nan

# Разделение признаков и целевых переменных
X = df.drop(['FinalScore', 'DroppedOut'], axis=1)

# Числовые и категориальные признаки
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

# Пайплайн для числовых признаков
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Пайплайн для категориальных признаков
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first'))
])

# Общий препроцессор
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Применение предобработки
X_processed = preprocessor.fit_transform(X)

# Получение названий признаков
feature_names_num = numeric_features
feature_names_cat = preprocessor.named_transformers_['cat'] \
    .named_steps['onehot'] \
    .get_feature_names_out(categorical_features)

all_feature_names = np.concatenate([feature_names_num, feature_names_cat])

# Преобразование в DataFrame
X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

# Результат
print("Исходная размерность:", X.shape)
print("После предобработки:", X_processed_df.shape)
print(X_processed_df.head())
