import os

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    """Load the trained Melbourne housing price model."""
    with st.spinner("Loading model..."):
        return load(model_path)


def get_user_input() -> pd.DataFrame:
    """Collect user input from the sidebar and return it as a DataFrame."""
    st.sidebar.header("🏡 Melbourne Property Details")

    type_ = st.sidebar.selectbox("Property Type", options=["h", "t", "u"])
    method = st.sidebar.selectbox(
        "Sale Method", options=["S", "SP", "PI", "VB", "SA", "SN", "W", "PN"]
    )
    suburb = st.sidebar.selectbox(
        "Suburb", options=["Abbotsford", "Northcote", "Richmond", "Carlton", "Fitzroy"]
    )
    regionname = st.sidebar.selectbox(
        "Region",
        options=[
            "Northern Metropolitan",
            "Southern Metropolitan",
            "Western Metropolitan",
            "Eastern Metropolitan",
        ],
    )
    councilarea = st.sidebar.selectbox(
        "Council Area", options=["Yarra", "Moreland", "Melbourne", "Darebin", "Port Phillip"]
    )

    rooms = st.sidebar.slider("Number of Rooms", min_value=1, max_value=10, value=3)
    bathroom = st.sidebar.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
    year_built = st.sidebar.slider("Year Built", min_value=1850, max_value=2025, value=1990)

    user_data = pd.DataFrame.from_dict(
        {
            "Type": [type_],
            "Method": [method],
            "Suburb": [suburb],
            "Rooms": [rooms],
            "Bathroom": [bathroom],
            "YearBuilt": [year_built],
            "Regionname": [regionname],
            "CouncilArea": [councilarea],
        }
    )
    return user_data


def main() -> None:
    st.set_page_config(page_title="Melbourne House Price Predictor 🏠", page_icon="📈")

    # Mostrar tu imagen local arriba del título
    #col1, col2, col3 = st.columns([1, 6, 1])
    #with col2:
        #st.image("melbournehouses.jpg", width=700)
    # st.image("melbournehouses.jpg")
    st.image("melbournehouses.jpg", use_column_width=True)


    st.title("📈 Melbourne Housing Price Estimator")
    st.write("Estimate the sale price of a property in Melbourne using a trained XGBoost model.")

    # Load model
    model_path = os.path.join("models", "melbourne_model.joblib")
    model = load_model(model_path)

    # User input
    input_df = get_user_input()

    # Prediction
    prediction = model.predict(input_df)[0]

    # Result display
    st.markdown("---")
    st.subheader("💰 Predicted Sale Price")
    st.success(f"Estimated Price: ${prediction:,.0f}")

    st.markdown("---")
    st.caption("Model: XGBoost | Data source: Melbourne Housing Dataset")


if __name__ == "__main__":
    main()
