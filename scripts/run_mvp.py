import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import cv2
from src.camera import Camera


def main():
    camera = Camera(source=0).open()
    print("Câmera aberta. Pressione 'q' para sair.")

    try:
        while True:
            frame = camera.read_frame()
            if frame is None:
                print("Falha ao ler frame da câmera.")
                break

            cv2.imshow("ElderGuard AI - Etapa 1", frame)

            # Encerra ao pressionar 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Câmera liberada. Encerrado com sucesso.")


if __name__ == "__main__":
    main()