from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

output_dir = Path("data/08_reporting/model_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

files = {
    "logistic_regression": "data/07_model_output/test_predictions.csv",
    "random_forest": "data/07_model_output/random_forest_test_predictions.csv",
}

for model_name, file_path in files.items():
    df = pd.read_csv(file_path)

    y_true = df["actual_target"].astype(int)
    y_pred = df["predicted_target"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Down (0)", "Up (1)"],
    )
    display.plot(ax=ax, colorbar=False)
    ax.set_title(f"{model_name.replace('_', ' ').title()} Test Confusion Matrix")

    output_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Created: {output_path}")
