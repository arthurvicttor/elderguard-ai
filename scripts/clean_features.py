import os
import argparse
import numpy as np
import pandas as pd

FPS = 25.0                # taxa de quadros do Le2i
MAX_GAP = 5               # buracos de até 5 frames (0,2 s) são interpolados
SMOOTH_WIN = 5            # janela da mediana móvel (causal)
REF_WIN = int(2 * FPS)    # janela p/ referência do tamanho do tronco
ASPECT = 320 / 240  # largura/altura do Le2i: converte x para a mesma unidade de y

META = ["scene", "video_file", "frame_idx", "frame_number",
        "phase", "is_fall", "pose_ok"]
OUT_COLS = META + ["hip_y", "trunk_len", "trunk_angle",
                   "hip_vy", "angle_vel", "valid"]


def fill_short_gaps(s, max_gap):
    """Interpola buracos curtos; buracos longos continuam NaN."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return filled.where(~(isna & (run_len > max_gap)))


def clean_series(s):
    filled = fill_short_gaps(s, MAX_GAP)
    smooth = filled.rolling(SMOOTH_WIN, min_periods=1).median()  # causal
    return smooth.where(filled.notna())  # não "vaza" valor para dentro de buracos


def clean_video(g):
    g = g.sort_values("frame_idx").copy()
    ok = g["pose_ok"] == 1

    raw = {
        "hip_x": (g["lm23_x"] + g["lm24_x"]) / 2 * ASPECT,
        "hip_y": (g["lm23_y"] + g["lm24_y"]) / 2,
        "sh_x": (g["lm11_x"] + g["lm12_x"]) / 2 * ASPECT,
        "sh_y": (g["lm11_y"] + g["lm12_y"]) / 2,
    }
    for name, s in raw.items():
        g[name] = clean_series(s.where(ok))

    dx = g["sh_x"] - g["hip_x"]
    dy = g["sh_y"] - g["hip_y"]
    g["trunk_len"] = np.hypot(dx, dy)
    # 0° = tronco vertical, 90° = horizontal (ignora de que lado fica a cabeça)
    g["trunk_angle"] = np.degrees(np.arctan2(dx.abs(), dy.abs()))

    # velocidade vertical em "tronco/segundo" (positivo = descendo)
    ref_len = g["trunk_len"].rolling(REF_WIN, min_periods=5).median()
    g["hip_vy"] = (g["hip_y"].diff() * FPS / ref_len).replace([np.inf, -np.inf], np.nan)
    g["angle_vel"] = g["trunk_angle"].diff() * FPS  # graus/segundo

    g["valid"] = g[["trunk_angle", "hip_vy", "angle_vel"]].notna().all(axis=1).astype(int)
    return g[OUT_COLS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/features_le2i.csv")
    parser.add_argument("--output", default="dataset/features_le2i_clean.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    parts = [clean_video(g) for _, g in df.groupby(["scene", "video_file"], sort=False)]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"Salvo em {args.output} ({len(out)} linhas)")

    print("\nFrames com features válidas, por fase:")
    print(out.groupby("phase")["valid"].mean().round(3))

    tmp = out[out["valid"] == 1].assign(
        abs_vy=lambda d: d["hip_vy"].abs(),
        abs_av=lambda d: d["angle_vel"].abs(),
    )
    cols = ["trunk_angle", "abs_vy", "abs_av"]
    print("\nMediana por fase:")
    print(tmp.groupby("phase")[cols].median().round(2))
    print("\nPercentil 90 por fase:")
    print(tmp.groupby("phase")[cols].quantile(0.9).round(2))


if __name__ == "__main__":
    main()