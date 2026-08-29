import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.camera import Camera
from src.pose import PoseEstimator
from src.features import TemporalFeatureExtractor
from src.events import FallDetector


def main():
    camera = Camera(source=0).open()
    pose_estimator = PoseEstimator()
    feature_extractor = TemporalFeatureExtractor()
    fall_detector = FallDetector()

    print("Iniciado. Pressione 'q' para sair.")
    prev_time = time.time()
    last_alert_text = None

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Falha ao ler frame da câmera.")
                break

            result = pose_estimator.process(frame)
            frame = pose_estimator.draw_landmarks(frame, result)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Filtro: só processa se ombros e quadris tiverem confiança mínima
                relevant_indices = [11, 12, 23, 24]  # ombros e quadris
                avg_visibility = sum(landmarks[i].visibility for i in relevant_indices) / len(relevant_indices)
                MIN_VISIBILITY = 0.5

                if avg_visibility >= MIN_VISIBILITY:
                    features = feature_extractor.extract(landmarks)
                    fall_status = fall_detector.update(features)

                    y0 = 60
                    for i, (name, value) in enumerate(features.items()):
                        cv2.putText(frame, f"{name}: {value:.3f}", (10, y0 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.putText(frame, f"state: {fall_status['state']}", (10, y0 + 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

                    if fall_status["event"] == "POSSIVEL_QUEDA":
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        last_alert_text = f"POSSIVEL QUEDA DETECTADA - {timestamp}"
                        print(f"⚠️ {last_alert_text}")
                else:
                    cv2.putText(frame, "Pessoa nao detectada com confianca suficiente", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if last_alert_text:
                cv2.putText(frame, last_alert_text, (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("ElderGuard AI - Etapa 4", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
        print("Encerrado com sucesso.")


if __name__ == "__main__":
    main()