import sys
import os
import time
import argparse
import warnings
from collections import deque
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import joblib
import numpy as np
import pandas as pd

from src.camera import Camera
from src.pose import PoseEstimator
import clean_features as cf
import make_windows as mw
from extract_le2i import CORE, MIN_VISIBILITY

BUFFER = 120                  # janela de 50 + 50 de referência do tronco + margem
EVAL_EVERY = mw.STRIDE        # avalia a cada 5 frames, como no treino
ALERT_SHOW_S = 8              # quanto tempo o alerta fica na tela
COOLDOWN_S = 10               # intervalo mínimo entre dois alertas
LM_COLS = [f"lm{i}_{c}" for i in CORE for c in ("x", "y", "vis")]
ROW_COLS = ["scene", "video_file", "frame_idx", "frame_number",
            "phase", "is_fall", "pose_ok"] + LM_COLS


def make_row(idx, lms):
    row = {"scene": "live", "video_file": "live", "frame_idx": idx,
           "frame_number": idx + 1, "phase": "live", "is_fall": 0, "pose_ok": 0}
    if lms is not None:
        vis = sum(lms[i].visibility for i in CORE) / len(CORE)
        for i in CORE:
            row[f"lm{i}_x"] = float(lms[i].x)
            row[f"lm{i}_y"] = float(lms[i].y)
            row[f"lm{i}_vis"] = float(lms[i].visibility)
        row["pose_ok"] = int(vis >= MIN_VISIBILITY)
    return row


def score_window(buffer, model, cols):
    """Mesma limpeza e mesmas features do treino. Retorna None se a janela não for válida."""
    df = pd.DataFrame(list(buffer)).reindex(columns=ROW_COLS)
    df = df.astype({c: float for c in LM_COLS})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cleaned = cf.clean_video(df)
        w = cleaned.iloc[-mw.WIN:]
        if len(w) < mw.WIN or w["valid"].mean() < mw.MIN_VALID_FRAC or w["valid"].iloc[-1] != 1:
            return None
        feats = mw.window_features(w)
    if feats is None or not np.all(np.isfinite(list(feats.values()))):
        return None
    return float(model.predict_proba(pd.DataFrame([feats])[cols])[0, 1])


def put(frame, text, y, color=(0, 255, 255), scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lr_all.joblib")
    parser.add_argument("--source", type=int, default=0)
    args = parser.parse_args()

    if not hasattr(cf, "ASPECT"):
        raise SystemExit("clean_features.py está sem a constante ASPECT (correção da proporção).")

    saved = joblib.load(args.model)
    model, cols, thr = saved["model"], saved["cols"], saved["threshold"]

    camera = Camera(source=args.source).open()
    pose = PoseEstimator()
    buffer = deque(maxlen=BUFFER)

    idx, fps, last_t = 0, None, None
    score, last_alert_t, alert_text, alert_until = None, -1e9, None, 0.0
    print(f"Iniciado. Limiar = {thr:.3f}. Pressione 'q' para sair.")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Falha ao ler frame da câmera.")
                break

            if idx == 0:
                h, w = frame.shape[:2]
                cf.ASPECT = w / h          # proporção real desta câmera
                print(f"Resolução {w}x{h}")

            now = time.perf_counter()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            last_t = now

            result = pose.process(frame)
            lms = result.pose_landmarks[0] if result.pose_landmarks else None
            frame = pose.draw_landmarks(frame, result)
            buffer.append(make_row(idx, lms))

            if len(buffer) >= mw.WIN and idx % EVAL_EVERY == 0:
                if fps:
                    cf.FPS = fps           # velocidades em unidades reais
                score = score_window(buffer, model, cols)
                t = time.time()
                if score is not None and score >= thr and t - last_alert_t > COOLDOWN_S:
                    last_alert_t = t
                    stamp = datetime.now().strftime("%H:%M:%S")
                    alert_text = f"POSSIVEL QUEDA DETECTADA - {stamp} (score {score:.2f})"
                    alert_until = t + ALERT_SHOW_S
                    print(f"[ALERTA] {alert_text}")

            if len(buffer) < mw.WIN:
                put(frame, "aquecendo...", 60)
            elif score is None:
                put(frame, "aguardando janela valida (pessoa visivel?)", 60)
            else:
                put(frame, f"score: {score:.2f}  (limiar {thr:.2f})", 60)

            if fps:
                color = (0, 255, 0) if fps >= 20 else (0, 0, 255)
                put(frame, f"FPS: {fps:.1f}", 30, color, 0.8)
                if fps < 20:
                    put(frame, "FPS baixo: resultados menos confiaveis", 85, (0, 0, 255))

            if alert_text and time.time() < alert_until:
                put(frame, alert_text, frame.shape[0] - 30, (0, 0, 255), 0.7)

            cv2.imshow("ElderGuard AI - Etapa 11 (ao vivo)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            idx += 1
    finally:
        camera.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()