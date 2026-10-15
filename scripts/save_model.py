import os
import joblib
import pandas as pd
from train_models import make_models, best_threshold

df = pd.read_csv("dataset/windows_le2i.csv")
train, val = df[df["split"] == "train"], df[df["split"] == "val"]

model, cols, _ = make_models()["lr_all"]
model.fit(train[cols], train["label"])
thr = best_threshold(val["label"], model.predict_proba(val[cols])[:, 1])

os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "cols": cols, "threshold": thr}, "models/lr_all.joblib")
print(f"Modelo salvo em models/lr_all.joblib | limiar = {thr:.3f} (Etapa 9: 0.795)")