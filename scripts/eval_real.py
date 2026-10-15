import argparse
import os
import cv2
import pandas as pd

import clean_features as cf
import make_windows as mw
from extract_le2i import process_video, COLUMNS
from train_models import make_models, best_threshold

NEGATIVE = ["normal", "walking", "sitting", "standing_up", "lying_down"]
CATEGORIES = NEGATIVE + ["falling"]
COMPARE = ["hip_drop", "angle_rise", "absvy_mean", "absav_max"]


def freeze_models(le2i_windows):
    """Mesmo procedimento da Etapa 9: treina no treino, limiar na validação."""
    df = pd.read_csv(le2i_windows)
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    frozen = {}
    for name, (model, cols, tune) in make_models().items():
        model.fit(train[cols], train["label"])
        thr = 0.5
        if tune:
            thr = best_threshold(val["label"], model.predict_proba(val[cols])[:, 1])
        frozen[name] = (model, cols, thr)
    return frozen, train


def load_annotations(path):
    if not os.path.exists(path):
        return {}
    a = pd.read_csv(path)
    return {r.video: (int(r.fall_start), int(r.fall_end)) for r in a.itertuples()}


def collect_clips(root, annotations):
    clips = []
    for cat in CATEGORIES:
        d = os.path.join(root, cat)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".mp4", ".avi", ".mov")):
                continue
            interval = None
            if cat == "falling":
                interval = annotations.get(f"{cat}/{f}")
                if interval is None:
                    print(f"[AVISO] sem anotação, ignorado: {cat}/{f}")
                    continue
            clips.append((cat, f, os.path.join(d, f), interval))
    return clips


def build_windows(clips, fps):
    if not hasattr(cf, "ASPECT"):
        raise SystemExit("clean_features.py está sem a constante ASPECT (correção da proporção).")
    rows = []
    for cat, f, path, interval in clips:
        cap = cv2.VideoCapture(path)
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        cf.ASPECT, cf.FPS = w / h, fps       # parâmetros desta câmera

        raw = process_video(path, cat, interval)
        if not raw:
            continue
        raw_df = pd.DataFrame(raw).reindex(columns=COLUMNS)
        cleaned = cf.clean_video(raw_df)
        r, _, _ = mw.make_windows(cleaned, cat, f)
        print(f"  {cat}/{f}: {int(w)}x{int(h)}, {len(raw)} frames, "
              f"pose ok {raw_df['pose_ok'].mean():.0%}, {len(r)} janelas")
        rows.extend(r)
    return pd.DataFrame(rows)


def neg_table(neg, models):
    out = {}
    for n in models:
        p, col = f"pred_{n}", {}
        for cat, g in neg.groupby("scene"):
            clips = g.groupby("clip")[p].max()
            col[cat] = f"{int(clips.sum())}/{len(clips)} clipes | {int(g[p].sum())}/{len(g)} janelas"
        out[n] = col
    return pd.DataFrame(out)


def fall_table(fall, models):
    pos, other = fall[fall["label"] == 1], fall[fall["label"] == 0]
    out = {}
    for n in models:
        p = f"pred_{n}"
        det = pos.groupby("clip")[p].max()
        out[n] = {"quedas detectadas": f"{int(det.sum())}/{len(det)}",
                  "alarmes fora da queda (janelas)": f"{int(other[p].sum())}/{len(other)}"}
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dataset")
    p.add_argument("--le2i", default="dataset/windows_le2i.csv")
    p.add_argument("--annotations", default="dataset/own_annotations.csv")
    p.add_argument("--fps", type=float, default=26.0)
    p.add_argument("--output", default="dataset/windows_own.csv")
    args = p.parse_args()

    frozen, train = freeze_models(args.le2i)
    print("Limiares (devem bater com a Etapa 9: rule 0.5, lr_angle 0.36, lr_all 0.795, rf_all 0.627):")
    print({n: round(t, 3) for n, (_, _, t) in frozen.items()})

    clips = collect_clips(args.root, load_annotations(args.annotations))
    print(f"\n{len(clips)} clipes (FPS assumido: {args.fps})")
    win = build_windows(clips, args.fps)
    if win.empty:
        print("Nenhuma janela gerada.")
        return

    win["clip"] = win["scene"] + "/" + win["video_file"]
    models = list(frozen)
    for n, (model, cols, thr) in frozen.items():
        win[f"pred_{n}"] = (model.predict_proba(win[cols])[:, 1] >= thr).astype(int)
    win.to_csv(args.output, index=False)

    pd.set_option("display.width", 220)
    print("\n=== ALARMES POR CLIPE (janelas) ===")
    print(win.groupby("clip").agg(
        janelas=("label", "size"), positivas=("label", "sum"),
        **{n: (f"pred_{n}", "sum") for n in models}))

    neg = win[win["scene"] != "falling"]
    fall = win[win["scene"] == "falling"]

    print("\n=== ATIVIDADES NORMAIS (qualquer alarme é falso alarme) ===")
    print(neg_table(neg, models))

    if not fall.empty:
        print("\n=== QUEDAS GRAVADAS (rotuladas por você) ===")
        print(fall_table(fall, models))

    print("\n=== DESLOCAMENTO DE DISTRIBUIÇÃO (mediana das features) ===")
    ref_neg = train[(train["label"] == 0) & (train["phase_end"] == "adl")][COMPARE].median()
    print("Negativos (atividades normais):")
    print(pd.DataFrame({"Le2i": ref_neg, "webcam": neg[COMPARE].median()}).round(2))
    pos_own = win[win["label"] == 1]
    if not pos_own.empty:
        ref_pos = train[train["label"] == 1][COMPARE].median()
        print("Positivos (quedas):")
        print(pd.DataFrame({"Le2i": ref_pos, "webcam": pos_own[COMPARE].median()}).round(2))


if __name__ == "__main__":
    main()