import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve

SEED = 42
FEATURES = ["angle_last", "angle_mean", "angle_max", "angle_rise",
            "vy_max", "absvy_mean", "absvy_recent",
            "absav_max", "absav_mean", "absav_recent", "hip_drop"]


class RuleBaseline:
    """Positivo se angle_rise >= a E hip_drop >= h. Limiares ajustados no TREINO."""

    def fit(self, X, y):
        best_f1, best = -1.0, (30, 0.5)
        for a in np.arange(10, 91, 5):
            for h in np.arange(0.1, 2.01, 0.1):
                pred = ((X["angle_rise"] >= a) & (X["hip_drop"] >= h)).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best = f1, (a, h)
        self.a, self.h = best
        return self

    def predict_proba(self, X):
        p = ((X["angle_rise"] >= self.a) & (X["hip_drop"] >= self.h)).astype(float).to_numpy()
        return np.column_stack([1 - p, p])


def make_models():
    return {
        "rule": (RuleBaseline(), ["angle_rise", "hip_drop"], False),
        "lr_angle": (make_pipeline(StandardScaler(),
                     LogisticRegression(class_weight="balanced", max_iter=1000)),
                     ["angle_last"], True),
        "lr_all": (make_pipeline(StandardScaler(),
                   LogisticRegression(class_weight="balanced", max_iter=1000)),
                   FEATURES, True),
        "rf_all": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                   class_weight="balanced", random_state=SEED, n_jobs=-1),
                   FEATURES, True),
    }


def best_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.nanargmax(f1[:-1])])


def evaluate(df, scores, thr):
    y = df["label"].to_numpy()
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    d = df.assign(pred=pred, vid=df["scene"] + "/" + df["video_file"])
    fall = d[d["label"] == 1].groupby("vid")["pred"].max()   # 1 = queda detectada
    fa_vids = d[(d["label"] == 0) & (d["pred"] == 1)]["vid"].nunique()

    return {
        "thr": round(thr, 3), "precision": round(prec, 3), "recall": round(rec, 3),
        "F1": round(f1, 3), "FPR": round(fp / (fp + tn), 3),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "quedas_detectadas": f"{int(fall.sum())}/{len(fall)}",
        "videos_c_alarme_falso": f"{fa_vids}/{d['vid'].nunique()}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/windows_le2i.csv")
    parser.add_argument("--test", choices=["rule", "lr_angle", "lr_all", "rf_all", "all"],
                        default=None,
                        help="Avalia no teste. Use uma única vez, no final.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"treino={len(train)}  val={len(val)}  teste={len(test)} janelas")

    results, fitted = {}, {}
    for name, (model, cols, tune) in make_models().items():
        model.fit(train[cols], train["label"])
        scores_val = model.predict_proba(val[cols])[:, 1]
        thr = best_threshold(val["label"], scores_val) if tune else 0.5
        results[name] = evaluate(val, scores_val, thr)
        fitted[name] = (model, cols, thr)

    pd.set_option("display.width", 200)
    print("\n=== VALIDAÇÃO (limiar escolhido aqui) ===")
    print(pd.DataFrame(results).T)

    rule = fitted["rule"][0]
    print(f"\nRegra aprendida no treino: angle_rise >= {rule.a} e hip_drop >= {rule.h:.1f}")

    coefs = fitted["lr_all"][0].named_steps["logisticregression"].coef_[0]
    print("\nCoeficientes da regressão logística (features padronizadas):")
    print(pd.Series(coefs, index=FEATURES).sort_values(key=abs, ascending=False).round(2))

    imp = fitted["rf_all"][0].feature_importances_
    print("\nImportância das features no Random Forest:")
    print(pd.Series(imp, index=FEATURES).sort_values(ascending=False).round(3))

    if args.test:
        names = list(fitted) if args.test == "all" else [args.test]
        rows = {}
        for n in names:
            model, cols, thr = fitted[n]
            rows[n] = evaluate(test, model.predict_proba(test[cols])[:, 1], thr)
        print("\n=== TESTE (limiares vindos da validação) ===")
        print(pd.DataFrame(rows).T)
        print("\nAtenção: o teste foi usado. Não ajuste mais nada com base nele.")


if __name__ == "__main__":
    main()