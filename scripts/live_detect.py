import sys
import os
import time
import argparse
import traceback
import warnings
from collections import deque
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
import joblib
import numpy as np
import pandas as pd

try:
    import psutil          # opcional: mede CPU e memória nos batimentos
except ImportError:
    psutil = None

from src.camera import Camera
from src.pose import PoseEstimator
from src.risk import RiskMonitor, Level
from src.alerts import EventLog
import clean_features as cf
import make_windows as mw
from extract_le2i import CORE, MIN_VISIBILITY

BUFFER = 120                  # janela de 50 + 50 de referência do tronco + margem
EVAL_EVERY = mw.STRIDE        # avalia a cada 5 frames, como no treino
ALERT_SHOW_S = 8              # quanto tempo o alerta de queda fica na tela
COOLDOWN_S = 10               # intervalo mínimo entre dois alertas do modelo
HEARTBEAT_S = 60              # um registro de funcionamento por minuto
GAP_S = 5.0                   # mais que isso entre dois frames = lacuna (ex: PC suspenso)
RECONNECT_WAIT_S = 2.0        # espera entre tentativas de reabrir a câmera

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
RISK_IDLE = {"score": 0, "level": Level.LOW, "reasons": [], "active": False, "resolved": False}


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
    parser.add_argument("--log-dir", default="logs", help="Pasta do log local de eventos")
    parser.add_argument("--no-log", action="store_true", help="Não registra eventos")
    parser.add_argument("--headless", action="store_true",
                        help="Sem janela nem desenho (Ctrl+C para parar)")
    parser.add_argument("--max-minutes", type=float, default=0,
                        help="Se > 0, encerra sozinho depois desse tempo (útil para testes)")
    parser.add_argument("--camera-timeout-s", type=float, default=300,
                        help="Encerra (código 2) se a câmera ficar sem sinal por esse tempo")
    args = parser.parse_args()

    if not hasattr(cf, "ASPECT"):
        raise SystemExit("clean_features.py está sem a constante ASPECT (correção da proporção).")

    saved = joblib.load(args.model)
    model, cols, thr = saved["model"], saved["cols"], saved["threshold"]

    proc = psutil.Process() if psutil else None
    if proc:
        proc.cpu_percent(None)     # a primeira chamada só inicia a medição

    camera = Camera(source=args.source).open()
    pose = PoseEstimator()
    buffer = deque(maxlen=BUFFER)
    monitor = RiskMonitor(immobile_s=args.immobile_s)
    log = EventLog(args.log_dir, enabled=not args.no_log)

    idx, fps, last_t, pose_ms = 0, None, None, 0.0
    score, feats, last_alert_t, alert_text, alert_until = None, None, -1e9, None, 0.0
    risk = dict(RISK_IDLE)
    last_level = Level.LOW
    t_start = time.time()
    last_hb = t_start
    last_wall = None
    need_aspect = True
    cam_lost_at = None
    reason, exit_code = "user_quit", 0

    log.log("session_start", threshold=round(thr, 3), immobile_s=args.immobile_s,
            width=args.width, source=args.source, headless=args.headless)
    stop_hint = "Ctrl+C para parar" if args.headless else "'q' sai, 'r' reinicia o episodio de risco"
    print(f"Iniciado. Limiar = {thr:.3f}. {stop_hint}.")
    if not args.no_log:
        print(f"Log local em '{args.log_dir}' (só números, nenhum vídeo é salvo).")
    if not proc:
        print("(psutil não instalado: CPU e memória não serão registradas)")

    try:
        while True:
            if args.max_minutes > 0 and time.time() - t_start > args.max_minutes * 60:
                reason = "max_minutes"
                break

            frame = camera.read_frame()
            if frame is None:
                if cam_lost_at is None:
                    cam_lost_at = time.time()
                    log.log("camera_lost")
                    print("[CAMERA] sem frames; tentando reconectar...")
                    buffer.clear()
                    last_t, last_wall, fps = None, None, None
                if time.time() - cam_lost_at > args.camera_timeout_s:
                    print("[CAMERA] sem sinal por tempo demais; encerrando.")
                    reason, exit_code = "camera_unrecoverable", 2
                    break
                camera.release()
                time.sleep(RECONNECT_WAIT_S)
                try:
                    camera.open()
                except RuntimeError:
                    pass
                continue

            if cam_lost_at is not None:
                down = time.time() - cam_lost_at
                log.log("camera_recovered", seconds=round(down, 1))
                print(f"[CAMERA] recuperada após {down:.0f}s")
                cam_lost_at = None
                need_aspect = True

            if args.width > 0 and frame.shape[1] > args.width:
                s = args.width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s)

            if need_aspect:
                h, w = frame.shape[:2]
                cf.ASPECT = w / h          # proporção real desta câmera
                print(f"Resolução {w}x{h}")
                log.log("camera_info", width=w, height=h)
                need_aspect = False

            wall = time.time()
            if last_wall is not None and wall - last_wall > GAP_S:
                gap = wall - last_wall
                log.log("time_gap", seconds=round(gap, 1))
                print(f"[TEMPO] lacuna de {gap:.0f}s (PC suspenso?); janela e episódio reiniciados")
                buffer.clear()
                monitor.reset()
                if last_level != Level.LOW:
                    log.log("risk_change", from_level=last_level.value,
                            to_level=Level.LOW.value, risk_score=0,
                            reasons=["reiniciado: lacuna de tempo"])
                risk, last_level = dict(RISK_IDLE), Level.LOW
                last_t = None
            last_wall = wall

            now = time.perf_counter()
            if last_t is not None:
                inst = 1.0 / max(now - last_t, 1e-6)
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            last_t = now

            t0 = time.perf_counter()
            result = pose.process(frame)
            pose_ms = 0.95 * pose_ms + 0.05 * (time.perf_counter() - t0) * 1000
            lms = result.pose_landmarks[0] if result.pose_landmarks else None
            if not args.headless:
                frame = pose.draw_landmarks(frame, result)
            buffer.append(make_row(idx, lms))

            if fps and wall - last_hb >= HEARTBEAT_S:
                last_hb = wall
                visible = sum(r["pose_ok"] for r in buffer) / max(len(buffer), 1)
                hb = dict(fps=round(fps, 1), pose_ms=round(pose_ms),
                          person_visible=round(visible, 2))
                if proc:
                    hb["cpu_pct"] = round(proc.cpu_percent(None), 1)
                    hb["rss_mb"] = round(proc.memory_info().rss / 1e6)
                log.log("heartbeat", **hb)
                print(f"[{datetime.now():%H:%M:%S}] fps {fps:.1f} | pessoa visível {visible:.0%} "
                      f"| risco {risk['level'].value}")

            if len(buffer) >= mw.WIN and idx % EVAL_EVERY == 0:
                if fps:
                    cf.FPS = fps           # velocidades em unidades reais
                cleaned = clean_buffer(buffer)
                score, feats = score_window(cleaned, model, cols)

                t = time.time()
                fall_alert = False
                if score is not None and score >= thr and t - last_alert_t > COOLDOWN_S:
                    last_alert_t = t
                    fall_alert = True
                    stamp = datetime.now().strftime("%H:%M:%S")
                    alert_text = f"POSSIVEL QUEDA DETECTADA - {stamp} (score {score:.2f})"
                    alert_until = t + ALERT_SHOW_S
                    print(f"[ALERTA] {alert_text}")
                    log.log("fall_alert", score=round(score, 3), threshold=round(thr, 3))

                pose_ok, upright, still = posture_state(cleaned)
                risk = monitor.update(t, fall_alert=fall_alert, pose_ok=pose_ok,
                                      on_ground_still=still, upright=upright)
                if risk["level"] != last_level:
                    print(f"[RISCO] {last_level.value} -> {risk['level'].value} "
                          f"({', '.join(risk['reasons'])})")
                    log.log("risk_change", from_level=last_level.value,
                            to_level=risk["level"].value, risk_score=risk["score"],
                            reasons=risk["reasons"])
                    last_level = risk["level"]

            if not args.headless:
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

                put(frame, f"RISCO: {risk['level'].value.upper()} ({risk['score']})",
                    140, LEVEL_COLOR[risk["level"]], 0.7)
                if risk["active"]:
                    put(frame, " / ".join(risk["reasons"]), 165, LEVEL_COLOR[risk["level"]])
                    if risk["level"] == Level.EMERGENCY:
                        put(frame, "VERIFIQUE A PESSOA", 195, (0, 0, 255), 0.8)

                if feats:
                    put(frame, f"hip_drop {feats['hip_drop']:.2f} | angle_rise {feats['angle_rise']:.0f} "
                               f"| absvy {feats['absvy_mean']:.2f}",
                        frame.shape[0] - 8, (255, 255, 0), 0.4)

                if alert_text and time.time() < alert_until:
                    put(frame, alert_text, frame.shape[0] - 30, (0, 0, 255), 0.7)

                cv2.imshow("ElderGuard AI - Etapa 14 (ao vivo)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    monitor.reset()
                    risk, last_level = dict(RISK_IDLE), Level.LOW
                    log.log("manual_reset")
                    print("[RISCO] episodio reiniciado manualmente")
            idx += 1

    except KeyboardInterrupt:
        reason = "interrupted"
        print("\nInterrompido pelo usuário.")
    except Exception as e:
        reason, exit_code = "crash", 1
        log.log("crash", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        log.log("session_end", frames=idx, reason=reason)
        camera.release()
        pose.close()
        cv2.destroyAllWindows()
        print("Encerrado.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()