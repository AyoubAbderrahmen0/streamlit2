# %% [1] Import des packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# %% [2] Charger les données
df = pd.read_csv("Financial_inclusion_dataset.csv")
df.drop_duplicates(inplace=True)

# %% [3] Remplacer valeurs manquantes
categorical_cols = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                    'relationship_with_head', 'marital_status', 'education_level', 'job_type']

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df['age_of_respondent'].fillna(df['age_of_respondent'].mean(), inplace=True)

# %% [4] Gestion des outliers
df['age_of_respondent'] = np.clip(df['age_of_respondent'], 16, 100)
df['household_size'] = np.clip(df['household_size'], 1, 20)

# %% [5] Feature Engineering simple
df['age_group'] = pd.cut(df['age_of_respondent'], bins=[15,30,50,100], labels=[0,1,2])
df['household_group'] = pd.cut(df['household_size'], bins=[0,3,6,20], labels=[0,1,2])

categorical_cols += ['age_group', 'household_group']

# %% [6] Encodage des variables catégoriques
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encoder la cible
target_le = LabelEncoder()
df['bank_account'] = target_le.fit_transform(df['bank_account'])

# %% [7] Préparation des données pour ML
X = df.drop(['uniqueid', 'bank_account', 'age_of_respondent', 'household_size'], axis=1)
y = df['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [8] RandomForest avec hyperparam tuning
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                               n_iter=20, cv=3, verbose=2, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)

best_model = rf_search.best_estimator_

# %% [9] Evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [10] Sauvegarder le modèle et les encoders
joblib.dump(best_model, "rf_model_optimized.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(target_le, "target_encoder.pkl")
print("Modèle et encoders sauvegardés !")
