import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.camera import Camera
from src.pose import PoseEstimator


def main():
    camera = Camera(source=0).open()
    pose_estimator = PoseEstimator()

    print("Câmera e pose estimator iniciados. Pressione 'q' para sair.")

    prev_time = time.time()

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Falha ao ler frame da câmera.")
                break

            results = pose_estimator.process(frame)
            frame = pose_estimator.draw_landmarks(frame, results)

            # Cálculo simples de FPS para observarmos a performance
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            cv2.imshow("ElderGuard AI - Etapa 2", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        pose_estimator.close()
        cv2.destroyAllWindows()
        print("Encerrado com sucesso.")


if __name__ == "__main__":
    main()