import cv2

class Camera:
    """Wrapper simples sobre a captura de vídeo da webcam."""

    def __init__(self, source: int = 0):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera.")
        return self

    def read_frame(self):
        """Lê um frame. Retorna None se falhar."""
        if self.cap is None:
            raise RuntimeError("Câmera não foi aberta. Chame open() primeiro.")
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()