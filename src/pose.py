import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class PoseEstimator:
    """Wrapper sobre a MediaPipe Tasks API (PoseLandmarker)."""

    def __init__(self, model_path: str = "models/pose_landmarker_lite.task"):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process(self, frame_bgr):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.timestamp_ms += 33  # incrementa timestamp (~30fps)
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        return result

    def draw_landmarks(self, frame_bgr, result):
        """Desenha os keypoints manualmente (a Tasks API não tem drawing_utils pronto)."""
        h, w, _ = frame_bgr.shape
        if result.pose_landmarks:
            for pose_landmarks in result.pose_landmarks:
                for landmark in pose_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)
        return frame_bgr

    def close(self):
        self.landmarker.close()