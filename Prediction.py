import sys
import joblib
import pandas as pd
from Preprocessing import DataQualityChecker


sys.modules["__main__"].DataQualityChecker = DataQualityChecker



MODEL_PATH = "model/dropout_pipeline_tuned.pkl"
INPUT_CSV = "data/data.csv"     
OUTPUT_CSV = "data/output_prediction.csv" 


def load_model(path: str):
    return joblib.load(path)


def main():

    print("Membaca data input...")
    df = pd.read_csv(INPUT_CSV, sep=";")

    if "Status" in df.columns:
        df = df.drop(columns=["Status"])

    print("Memuat model...")
    model = load_model(MODEL_PATH)

    print("Melakukan prediksi...")
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)

    label_map = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }

    df["Predicted_Status"] = [label_map[p] for p in predictions]
    df["Prob_Dropout"] = probabilities[:, 0]
    df["Prob_Enrolled"] = probabilities[:, 1]
    df["Prob_Graduate"] = probabilities[:, 2]

    print("Menyimpan hasil ke CSV...")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Prediksi selesai!")
    print(f"Hasil disimpan di: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
