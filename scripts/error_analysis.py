import pandas as pd
from train_models import make_models, best_threshold

FPS = 25
df = pd.read_csv("dataset/windows_le2i.csv")
train = df[df["split"] == "train"]
val = df[df["split"] == "val"].copy()
val["vid"] = val["scene"] + "/" + val["video_file"]
val["video_com_queda"] = val["vid"].map(val.groupby("vid")["label"].max())

for name in ["rule", "lr_all", "rf_all"]:
    model, cols, tune = make_models()[name]
    model.fit(train[cols], train["label"])
    score = model.predict_proba(val[cols])[:, 1]
    thr = best_threshold(val["label"], score) if tune else 0.5
    val["pred"] = (score >= thr).astype(int)

    print(f"\n===== {name} =====")

    fp = val[(val["label"] == 0) & (val["pred"] == 1)]
    print("Falsos positivos (fase onde a janela termina x tipo de vídeo):")
    print(pd.crosstab(fp["phase_end"],
                      fp["video_com_queda"].map({0: "sem queda", 1: "com queda"})))

    fn = val[(val["label"] == 1) & (val["pred"] == 0)]
    print("Falsos negativos por fase onde a janela termina:")
    print(fn["phase_end"].value_counts().to_string())

    lat = []
    for vid, g in val[val["label"] == 1].groupby("vid"):
        hits = g[g["pred"] == 1]
        if len(hits):
            lat.append((hits["end_frame_number"].min() - g["end_frame_number"].min()) / FPS)
    print(f"Atraso do 1º alarme após a 1ª janela positiva (s): "
          f"mediana={pd.Series(lat).median():.1f}, máx={max(lat):.1f}")