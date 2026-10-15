import argparse
import hashlib
import warnings
import numpy as np
import pandas as pd

FPS = 25
WIN = 50              # 2 s
STRIDE = 5            # 0,2 s
RECENT = 12           # ~0,5 s finais da janela
MIN_VALID_FRAC = 0.8
MIN_OVERLAP = 10      # frames de queda na janela p/ ser positiva
SPLIT_SALT = "v1"
TRAIN_PCT, VAL_PCT = 70, 15   # o resto vai para teste

FEATURES = ["angle_last", "angle_mean", "angle_max", "angle_rise",
            "vy_max", "absvy_mean", "absvy_recent",
            "absav_max", "absav_mean", "absav_recent", "hip_drop"]


def assign_split(scene, video_file):
    key = f"{SPLIT_SALT}/{scene}/{video_file}".encode()
    bucket = int(hashlib.md5(key).hexdigest(), 16) % 100
    if bucket < TRAIN_PCT:
        return "train"
    if bucket < TRAIN_PCT + VAL_PCT:
        return "val"
    return "test"


def window_features(w):
    ang = w["trunk_angle"].to_numpy()
    vy = w["hip_vy"].to_numpy()
    av = w["angle_vel"].to_numpy()
    hip = w["hip_y"].to_numpy()
    ref = np.nanmedian(w["trunk_len"].to_numpy())
    if not np.isfinite(ref) or ref <= 0:
        return None
    return {
        "angle_last": ang[-1],
        "angle_mean": np.nanmean(ang),
        "angle_max": np.nanmax(ang),
        "angle_rise": ang[-1] - np.nanmin(ang),     # quanto o tronco inclinou na janela
        "vy_max": np.nanmax(vy),                    # maior velocidade de descida
        "absvy_mean": np.nanmean(np.abs(vy)),
        "absvy_recent": np.nanmean(np.abs(vy[-RECENT:])),   # ~0 = parado agora
        "absav_max": np.nanmax(np.abs(av)),
        "absav_mean": np.nanmean(np.abs(av)),
        "absav_recent": np.nanmean(np.abs(av[-RECENT:])),
        "hip_drop": (hip[-1] - np.nanmin(hip)) / ref,       # queda do quadril, em "troncos"
    }


def make_windows(g, scene, video_file):
    g = g.sort_values("frame_idx").reset_index(drop=True)
    fall_len = int(g["is_fall"].sum())
    need = min(MIN_OVERLAP, fall_len)
    rows, skipped_invalid, skipped_ambiguous = [], 0, 0

    for end in range(WIN - 1, len(g), STRIDE):
        w = g.iloc[end - WIN + 1: end + 1]

        overlap = int(w["is_fall"].sum())
        if overlap == 0:
            label = 0
        elif overlap >= need:
            label = 1
        else:
            skipped_ambiguous += 1
            continue

        if w["valid"].mean() < MIN_VALID_FRAC or w["valid"].iloc[-1] != 1:
            skipped_invalid += 1
            continue

        feats = window_features(w)
        if feats is None or not np.all(np.isfinite(list(feats.values()))):
            skipped_invalid += 1
            continue

        rows.append({
            "scene": scene, "video_file": video_file,
            "split": assign_split(scene, video_file),
            "end_frame_number": int(w["frame_number"].iloc[-1]),
            "phase_end": w["phase"].iloc[-1],
            "overlap": overlap, "label": label, **feats,
        })
    return rows, skipped_invalid, skipped_ambiguous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/features_le2i_clean.csv")
    parser.add_argument("--output", default="dataset/windows_le2i.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    all_rows, inv, amb = [], 0, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # janelas com NaN parcial
        for (scene, vf), g in df.groupby(["scene", "video_file"], sort=False):
            rows, i, a = make_windows(g, scene, vf)
            all_rows.extend(rows)
            inv += i
            amb += a

    out = pd.DataFrame(all_rows)
    out.to_csv(args.output, index=False)
    out["video_id"] = out["scene"] + "/" + out["video_file"]

    print(f"Janelas geradas: {len(out)}  (salvo em {args.output})")
    print(f"Descartadas: {inv} por frames inválidos, {amb} ambíguas (sobreposição parcial)")

    summary = out.groupby("split").agg(
        videos=("video_id", "nunique"),
        windows=("label", "size"),
        positives=("label", "sum"),
    )
    summary["pos_rate"] = (summary["positives"] / summary["windows"]).round(3)
    summary["videos_com_queda"] = out[out["label"] == 1].groupby("split")["video_id"].nunique()
    summary = summary.reindex(["train", "val", "test"]).fillna(0)
    print("\nResumo por grupo:")
    print(summary)

    fall_vids = df.loc[df["is_fall"] == 1, ["scene", "video_file"]].drop_duplicates()
    fall_ids = set(fall_vids["scene"] + "/" + fall_vids["video_file"])
    lost = sorted(fall_ids - set(out.loc[out["label"] == 1, "video_id"]))
    print(f"\nVídeos com queda sem nenhuma janela positiva: {len(lost)}")
    if lost:
        print(lost[:10])

    print("\nMediana das features por rótulo (0 = sem queda, 1 = queda):")
    print(out.groupby("label")[FEATURES].median().round(2).T)


if __name__ == "__main__":
    main()