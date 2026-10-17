import sys
import os
import time
import argparse
import warnings
from collections import deque
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import joblib
import numpy as np
import pandas as pd

from src.camera import Camera
from src.pose import PoseEstimator
from risk import RiskMonitor, Level, level_for
import clean_features as cf
import make_windows as mw
from extract_le2i import CORE, MIN_VISIBILITY

BUFFER = 120                  # janela de 50 + 50 de referência do tronco + margem
EVAL_EVERY = mw.STRIDE        # avalia a cada 5 frames, como no treino
ALERT_SHOW_S = 8              # quanto tempo o alerta de queda fica na tela
COOLDOWN_S = 10               # intervalo mínimo entre dois alertas do modelo

# Postura/imobilidade (valores INICIAIS, a calibrar nos testes)
STILL_FRAMES = 25             # ~1 s de frames recentes
ANGLE_GROUND = 40.0           # tronco >= 40° = "no chão"
ANGLE_UPRIGHT = 30.0          # tronco < 30° = "de pé"
STILL_VY = 0.10               # |velocidade do quadril| mediana (troncos/s)
STILL_AV = 10.0               # |velocidade do ângulo| mediana (graus/s)

LM_COLS = [f"lm{i}_{c}" for i in CORE for c in ("x", "y", "vis")]
ROW_COLS = ["scene", "video_file", "frame_idx", "frame_number",
            "phase", "is_fall", "pose_ok"] + LM_COLS
LEVEL_COLOR = {Level.LOW: (0, 255, 0), Level.ATTENTION: (0, 255, 255),
               Level.ALERT: (0, 165, 255), Level.EMERGENCY: (0, 0, 255)}


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


def clean_buffer(buffer):
    """Mesma limpeza do treino, aplicada uma vez por avaliação."""
    df = pd.DataFrame(list(buffer)).reindex(columns=ROW_COLS)
    df = df.astype({c: float for c in LM_COLS})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return cf.clean_video(df)


def score_window(cleaned, model, cols):
    """Retorna (score, features) da janela atual, ou (None, None) se a janela não for válida."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w = cleaned.iloc[-mw.WIN:]
        if len(w) < mw.WIN or w["valid"].mean() < mw.MIN_VALID_FRAC or w["valid"].iloc[-1] != 1:
            return None, None
        feats = mw.window_features(w)
    if feats is None or not np.all(np.isfinite(list(feats.values()))):
        return None, None
    score = float(model.predict_proba(pd.DataFrame([feats])[cols])[0, 1])
    return score, feats


def posture_state(cleaned):
    """Retorna (pose_ok, upright, on_ground_still) olhando ~1 s recente."""
    r = cleaned.iloc[-STILL_FRAMES:]
    if r["trunk_angle"].notna().mean() < 0.5:
        return False, False, False
    ang = r["trunk_angle"].median()
    vy = r["hip_vy"].abs().median()
    av = r["angle_vel"].abs().median()
    upright = bool(ang < ANGLE_UPRIGHT)
    still = bool(ang >= ANGLE_GROUND and vy < STILL_VY and av < STILL_AV)
    return True, upright, still


def put(frame, text, y, color=(0, 255, 255), scale=0.6):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/lr_all.joblib")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--width", type=int, default=0,
                        help="Se > 0, reduz o frame para essa largura (ex: 320) para ganhar FPS")
    parser.add_argument("--immobile-s", type=float, default=10.0,
                        help="Segundos parado no chão para subir o risco (10 = valor de teste)")
    args = parser.parse_args()

    if not hasattr(cf, "ASPECT"):
        raise SystemExit("clean_features.py está sem a constante ASPECT (correção da proporção).")

    saved = joblib.load(args.model)
    model, cols, thr = saved["model"], saved["cols"], saved["threshold"]

    camera = Camera(source=args.source).open()
    pose = PoseEstimator()
    buffer = deque(maxlen=BUFFER)
    monitor = RiskMonitor(immobile_s=args.immobile_s)

    idx, fps, last_t, pose_ms = 0, None, None, 0.0
    score, feats, last_alert_t, alert_text, alert_until = None, -1e9, None, 0.0
    risk = {"score": 0, "level": Level.LOW, "reasons": [], "active": False, "resolved": False}
    last_level = Level.LOW
    print(f"Iniciado. Limiar = {thr:.3f}. 'q' sai, 'r' reinicia o episodio de risco.")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Falha ao ler frame da câmera.")
                break

            if args.width > 0 and frame.shape[1] > args.width:
                s = args.width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s)

            if idx == 0:
                h, w = frame.shape[:2]
                cf.ASPECT = w / h          # proporção real desta câmera
                print(f"Resolução {w}x{h}")

            now = time.perf_counter()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            last_t = now

            t0 = time.perf_counter()
            result = pose.process(frame)
            pose_ms = 0.95 * pose_ms + 0.05 * (time.perf_counter() - t0) * 1000
            lms = result.pose_landmarks[0] if result.pose_landmarks else None
            frame = pose.draw_landmarks(frame, result)
            buffer.append(make_row(idx, lms))

            if len(buffer) >= mw.WIN and idx % EVAL_EVERY == 0:
                if fps:
                    cf.FPS = fps           # velocidades em unidades reais
                cleaned = clean_buffer(buffer)
                score, feats= score_window(cleaned, model, cols)

                t = time.time()
                fall_alert = False
                if score is not None and score >= thr and t - last_alert_t > COOLDOWN_S:
                    last_alert_t = t
                    fall_alert = True
                    stamp = datetime.now().strftime("%H:%M:%S")
                    alert_text = f"POSSIVEL QUEDA DETECTADA - {stamp} (score {score:.2f})"
                    alert_until = t + ALERT_SHOW_S
                    print(f"[ALERTA] {alert_text}")

                pose_ok, upright, still = posture_state(cleaned)
                risk = monitor.update(t, fall_alert=fall_alert, pose_ok=pose_ok,
                                      on_ground_still=still, upright=upright)
                if risk["level"] != last_level:
                    print(f"[RISCO] {last_level.value} -> {risk['level'].value} "
                          f"({', '.join(risk['reasons'])})")
                    last_level = risk["level"]

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
            put(frame, f"pose: {pose_ms:.0f} ms/frame", 110)

            if feats:
                put(frame, f"hip_drop {feats['hip_drop']:.2f} | angle_rise {feats['angle_rise']:.0f} | absvy {feats['absvy_mean']:.2f}",
                    frame.shape[0] - 8, (255, 255, 0), 0.4)

            put(frame, f"RISCO: {risk['level'].value.upper()} ({risk['score']})",
                140, LEVEL_COLOR[risk["level"]], 0.7)
            if risk["active"]:
                put(frame, " / ".join(risk["reasons"]), 165, LEVEL_COLOR[risk["level"]])
                if risk["level"] == Level.EMERGENCY:
                    put(frame, "VERIFIQUE A PESSOA", 195, (0, 0, 255), 0.8)

            if alert_text and time.time() < alert_until:
                put(frame, alert_text, frame.shape[0] - 30, (0, 0, 255), 0.7)

            cv2.imshow("ElderGuard AI - Etapa 12 (ao vivo)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                monitor.reset()
                risk = {"score": 0, "level": Level.LOW, "reasons": [], "active": False, "resolved": False}
                last_level = Level.LOW
                print("[RISCO] episodio reiniciado manualmente")
            idx += 1
    finally:
        camera.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()