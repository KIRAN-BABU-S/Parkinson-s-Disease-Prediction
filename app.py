import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("parkinsons_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.title("Parkinson's Disease Prediction")
st.write("Enter all the required medical parameters to check if a person has Parkinson's disease.")


feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

user_inputs = []

for name in feature_names:
    val = st.number_input(f"Enter {name}:", format="%.6f")
    user_inputs.append(val)


if st.button("Predict"):
    try:
        
        input_data_np = np.asarray(user_inputs).reshape(1, -1)

        
        std_data = scaler.transform(input_data_np)

        
        prediction = model.predict(std_data)

        
        if prediction[0] == 0:
            st.success("The person does NOT have Parkinson's disease.")
        else:
            st.error("The person HAS Parkinson's disease.")
    except Exception as e:
        st.error(f"Error: {e}")
