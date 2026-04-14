import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

REF_PATH = Path("models/reference_data.csv")
CUR_PATH = Path("logs/inference_log.csv")
OUT_PATH = Path("reports/drift_report.html")

def run():
    if not REF_PATH.exists():
        print("Error: reference_data.csv not found")
        return

    if not CUR_PATH.exists():
        print("Error: inference_log.csv not found")
        return

    reference = pd.read_csv(REF_PATH)
    current = pd.read_csv(CUR_PATH)

    if len(current) < 50:
        print("Warning: not enough data for drift detection (< 50 rows).")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Mengkalkulasi data drift...")
    
    # 1. Definisikan report
    report_definition = Report(metrics=[DataDriftPreset()])
    
    # 2. TANGKAP HASIL EVALUASI KE DALAM VARIABEL BARU
    evaluation_result = report_definition.run(reference_data=reference, current_data=current)

    # 3. Panggil method save_html() dari objek hasil evaluasi
    evaluation_result.save_html(str(OUT_PATH))
    
    print(f"Sukses! Drift report generated at: {OUT_PATH}")

if __name__ == "__main__":
    run()