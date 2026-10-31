import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Diabetes Indicators Explorer",
    page_icon="🩺",
    layout="wide"
)

st.title("Diabetes Indicators Explorer")
st.caption("Explore diabetes indicators using a Machine Learning model.")

st.info("Educational project only. This application does not provide medical diagnoses.")


def yes_no(label):
    return st.selectbox(
        label,
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


# Dataset category labels
age_options = {
    1: "18–24", 2: "25–29", 3: "30–34",
    4: "35–39", 5: "40–44", 6: "45–49",
    7: "50–54", 8: "55–59", 9: "60–64",
    10: "65–69", 11: "70–74", 12: "75–79",
    13: "80 or older"
}

education_options = {
    1: "Never attended / Kindergarten",
    2: "Grades 1–8",
    3: "Grades 9–11",
    4: "High school graduate / GED",
    5: "College 1–3 years",
    6: "College 4+ years"
}

income_options = {
    1: "Below $10,000",
    2: "$10,000–$14,999",
    3: "$15,000–$19,999",
    4: "$20,000–$24,999",
    5: "$25,000–$34,999",
    6: "$35,000–$49,999",
    7: "$50,000–$74,999",
    8: "$75,000 or more"
}

health_options = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}


with st.form("diabetes_form"):

    col1, col2, col3 = st.columns(3)

    # COLUMN 1
    with col1:
        st.subheader("Health Indicators")

        HighBP = yes_no("High Blood Pressure")
        HighChol = yes_no("High Cholesterol")
        CholCheck = yes_no("Cholesterol Check")
        Smoker = yes_no("Smoked at least 100 cigarettes")
        Stroke = yes_no("History of Stroke")
        HeartDiseaseorAttack = yes_no("Heart Disease or Heart Attack")
        PhysActivity = yes_no("Physical Activity")

    # COLUMN 2
    with col2:
        st.subheader("Lifestyle & Other Factors")

        Fruits = yes_no("Consumes Fruits")
        Veggies = yes_no("Consumes Vegetables")
        HvyAlcoholConsump = yes_no("Heavy Alcohol Consumption")
        AnyHealthcare = yes_no("Has Healthcare Coverage")
        NoDocbcCost = yes_no("Could Not Visit Doctor Due to Cost")
        DiffWalk = yes_no("Difficulty Walking")
        Sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

    # COLUMN 3
    with col3:
        st.subheader("Personal Information")

        BMI = st.number_input(
            "BMI", min_value=12, max_value=98, value=27
        )

        MentHlth = st.number_input(
            "Mental Health (days)", 0, 30, 0
        )

        PhysHlth = st.number_input(
            "Physical Health (days)", 0, 30, 0
        )

        GenHlth = st.selectbox(
            "General Health",
            list(health_options),
            format_func=health_options.get
        )

        Age = st.selectbox(
            "Age Group",
            list(age_options),
            format_func=age_options.get
        )

        Education = st.selectbox(
            "Education",
            list(education_options),
            format_func=education_options.get
        )

        Income = st.selectbox(
            "Income",
            list(income_options),
            format_func=income_options.get
        )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )


if submitted:

    inputs = {
        "HighBP": HighBP,
        "HighChol": HighChol,
        "CholCheck": CholCheck,
        "BMI": BMI,
        "Smoker": Smoker,
        "Stroke": Stroke,
        "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth,
        "MentHlth": MentHlth,
        "PhysHlth": PhysHlth,
        "DiffWalk": DiffWalk,
        "Sex": Sex,
        "Age": Age,
        "Education": Education,
        "Income": Income
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=inputs,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("Prediction Result")

        if result["prediction"] == 1:
            st.error("Predicted Class: 1")
        else:
            st.success("Predicted Class: 0")

        st.metric(
            "Predicted Probability (Class 1)",
            f'{result["probability"]:.2%}'
        )

        st.caption(
            f'Classification threshold: {result["threshold"]:.2f}'
        )

        st.caption(
            "The probability is a model output, not a clinically "
            "validated estimate of diabetes risk."
        )

    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {e}")


st.divider()

st.header("Batch Prediction")
st.write("Upload a CSV file to predict multiple records.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

required_features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    missing = [
        col for col in required_features
        if col not in df.columns
    ]

    if missing:
        st.error(f"Missing columns: {missing}")

    elif df[required_features].isna().any().any():
        st.error("The CSV contains missing values.")

    else:
        st.success(f"{len(df)} records loaded successfully.")

        st.dataframe(df[required_features].head())

        if st.button("Predict All Records"):

            results = []

            try:
                for _, row in df[required_features].iterrows():

                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        json=row.to_dict(),
                        timeout=30
                    )

                    response.raise_for_status()
                    results.append(response.json())

                output = df[required_features].copy()

                output["Prediction"] = [
                    r["prediction"] for r in results
                ]

                output["Probability"] = [
                    r["probability"] for r in results
                ]

                output["Threshold"] = [
                    r["threshold"] for r in results
                ]

                st.subheader("Batch Prediction Results")
                st.dataframe(output, use_container_width=True)

                st.download_button(
                    "Download Predictions",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="diabetes_predictions.csv",
                    mime="text/csv"
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Prediction failed: {e}")
