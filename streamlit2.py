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
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# %% [3] Vérifier doublons et valeurs manquantes
df.drop_duplicates(inplace=True)
print("Valeurs manquantes par colonne:\n", df.isnull().sum())

# %% [4] Remplacer valeurs manquantes
categorical_cols = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                    'relationship_with_head', 'marital_status', 'education_level', 'job_type']

for col in categorical_cols:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

# Colonne numérique
df['age_of_respondent'].fillna(df['age_of_respondent'].mean(), inplace=True)

# %% [5] Gestion des outliers
df['age_of_respondent'] = np.clip(df['age_of_respondent'], 16, 100)
df['household_size'] = np.clip(df['household_size'], 1, 20)

# %% [6] Feature Engineering simple
# Grouper l'age
df['age_group'] = pd.cut(df['age_of_respondent'], bins=[15,30,50,100], labels=[0,1,2])
# Taille ménage
df['household_group'] = pd.cut(df['household_size'], bins=[0,3,6,20], labels=[0,1,2])

categorical_cols += ['age_group', 'household_group']

# %% [7] Encodage des variables catégoriques
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encoder la cible
target_le = LabelEncoder()
df['bank_account'] = target_le.fit_transform(df['bank_account'])

# %% [8] Préparation des données pour ML
X = df.drop(['uniqueid', 'bank_account', 'age_of_respondent', 'household_size'], axis=1)
y = df['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [9] RandomForest avec hyperparam tuning
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

# %% [10] Evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %% [11] Importance des features
importances = best_model.feature_importances_
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importances")
plt.show()

# %% [12] Sauvegarder le modèle
joblib.dump(best_model, "rf_model_optimized.pkl")
print("Modèle et encoders sauvegardés !")

#%%
import os
print(os.path.getsize("rf_model_optimized.pkl") / 1024**2, "MB")