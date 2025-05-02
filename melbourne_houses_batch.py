import os

import pandas as pd
import streamlit as st
from joblib import load
from sklearn.pipeline import Pipeline


def load_model(model_path: str) -> Pipeline:
    with st.spinner("Loading model..."):
        return load(model_path)


def preprocess_melbourne_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Asegúrate de que las columnas sean del tipo correcto
    df["Rooms"] = pd.to_numeric(df["Rooms"], errors="coerce")
    df["Bathroom"] = pd.to_numeric(df["Bathroom"], errors="coerce")
    df["YearBuilt"] = pd.to_numeric(df["YearBuilt"], errors="coerce")
    return df


def batch_prediction_tab(model: Pipeline) -> None:
    st.subheader("📤 Upload your CSV file with Melbourne property data")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("🔍 Preview of uploaded data:")
            st.dataframe(df.head())

            required_cols = [
                "Type",
                "Method",
                "Suburb",
                "Rooms",
                "Bathroom",
                "YearBuilt",
                "Regionname",
                "CouncilArea",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}")
            elif st.button("Predict Prices"):
                df_clean = preprocess_melbourne_data(df)
                predictions = model.predict(df_clean)

                df_result = df.copy()
                df_result["Predicted Price"] = predictions.astype(int)

                st.success("✅ Predictions completed!")
                st.dataframe(df_result)

                st.download_button(
                    label="💾 Download results as CSV",
                    data=df_result.to_csv(index=False),
                    file_name="melbourne_price_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")
    else:
        st.info("Please upload a CSV file with Melbourne property information.")
        st.caption("Sample format:")
        sample_data = pd.DataFrame(
            {
                "Type": ["h", "u"],
                "Method": ["S", "PI"],
                "Suburb": ["Abbotsford", "Fitzroy"],
                "Rooms": [3, 2],
                "Bathroom": [1, 2],
                "YearBuilt": [1990, 2005],
                "Regionname": ["Northern Metropolitan", "Eastern Metropolitan"],
                "CouncilArea": ["Yarra", "Melbourne"],
            }
        )
        st.dataframe(sample_data)


def main() -> None:
    st.set_page_config(page_title="Batch Melbourne House Price Predictor", page_icon="🏠")
    st.image("melbournehouses.jpg", width=700)
    st.title("📦 Batch Melbourne Housing Price Estimator")
    st.write("Upload a CSV file to estimate sale prices of Melbourne properties.")

    model_path = os.path.join("models", "melbourne_model.joblib")
    model = load_model(model_path)

    batch_prediction_tab(model)


if __name__ == "__main__":
    main()
