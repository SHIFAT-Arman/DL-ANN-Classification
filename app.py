import streamlit as st
import keras
import pandas as pd
import pickle


# Load the model
model = keras.saving.load_model("model.keras")

# load the scaler and encoder
with open("scaler.pkl", "rb") as f:  # feature scaling
    scaler = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    encoder_geo = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)


# streamlit app
st.title('Customer Churn Prediction')

# user input
geography = st.selectbox('Geography', encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


# prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
})

# one-hot encoding for 'Geography'
geo_encoded = encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(['Geography']))

# combine the input data and the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# feature scaling
input_data_scaled = scaler.transform(input_data)

# make prediction churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f'The customer is likely to churn with probability {prediction_proba:.2f}')
else:
    st.write(f'The customer is not likely to churn with probability  {prediction_proba:.2f}')
