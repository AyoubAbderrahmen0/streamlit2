# %% [1] Import des packages
import streamlit as st
import pandas as pd
import joblib
import os

# %% [2] Vérifier l'existence des fichiers
required_files = ["rf_model_optimized.pkl", "label_encoders.pkl", "target_encoder.pkl"]
for f in required_files:
    if not os.path.exists(f):
        st.error(f"Fichier manquant : {f}")
        st.stop()

# %% [3] Charger le modèle et les encoders
model = joblib.load("rf_model_optimized.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# %% [4] Interface utilisateur Streamlit
st.title("Prédiction d'un compte bancaire")

with st.form("user_input"):
    country = st.selectbox("Pays", list(label_encoders['country'].classes_))
    year = st.number_input("Année", min_value=2016, max_value=2018, value=2017)
    location_type = st.selectbox("Type de localisation", list(label_encoders['location_type'].classes_))
    cellphone_access = st.selectbox("Accès téléphone", list(label_encoders['cellphone_access'].classes_))
    household_size = st.number_input("Taille du foyer", min_value=1, max_value=21, value=3)
    age_of_respondent = st.number_input("Âge du répondant", min_value=16, max_value=100, value=35)
    gender_of_respondent = st.selectbox("Genre", list(label_encoders['gender_of_respondent'].classes_))
    relationship_with_head = st.selectbox("Relation avec le chef du ménage",
                                          list(label_encoders['relationship_with_head'].classes_))
    marital_status = st.selectbox("Statut marital", list(label_encoders['marital_status'].classes_))
    education_level = st.selectbox("Niveau d'éducation", list(label_encoders['education_level'].classes_))
    job_type = st.selectbox("Type d'emploi", list(label_encoders['job_type'].classes_))

    submitted = st.form_submit_button("Prédire")

# %% [5] Faire la prédiction
if submitted:
    # Grouper l'age et la taille du ménage comme pendant l'entraînement
    if age_of_respondent <= 30:
        age_group = 0
    elif age_of_respondent <= 50:
        age_group = 1
    else:
        age_group = 2

    if household_size <= 3:
        household_group = 0
    elif household_size <= 6:
        household_group = 1
    else:
        household_group = 2

    input_data = pd.DataFrame({
        'country': [label_encoders['country'].transform([country])[0]],
        'year': [year],
        'location_type': [label_encoders['location_type'].transform([location_type])[0]],
        'cellphone_access': [label_encoders['cellphone_access'].transform([cellphone_access])[0]],
        'gender_of_respondent': [label_encoders['gender_of_respondent'].transform([gender_of_respondent])[0]],
        'relationship_with_head': [label_encoders['relationship_with_head'].transform([relationship_with_head])[0]],
        'marital_status': [label_encoders['marital_status'].transform([marital_status])[0]],
        'education_level': [label_encoders['education_level'].transform([education_level])[0]],
        'job_type': [label_encoders['job_type'].transform([job_type])[0]],
        'age_group': [age_group],
        'household_group': [household_group]
    })

    prediction = model.predict(input_data)
    prediction_label = target_encoder.inverse_transform(prediction)[0]
    st.success(f"Le répondant est susceptible d'avoir un compte bancaire: {prediction_label}")
