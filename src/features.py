import numpy as np


# Índices dos landmarks relevantes (MediaPipe Pose)
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24


def _landmark_to_xy(landmark):
    return np.array([landmark.x, landmark.y])


def get_hip_center(pose_landmarks):
    """Retorna o ponto médio entre os dois quadris (coords normalizadas 0-1)."""
    left_hip = _landmark_to_xy(pose_landmarks[LEFT_HIP])
    right_hip = _landmark_to_xy(pose_landmarks[RIGHT_HIP])
    return (left_hip + right_hip) / 2.0


def get_shoulder_center(pose_landmarks):
    """Retorna o ponto médio entre os dois ombros (coords normalizadas 0-1)."""
    left_shoulder = _landmark_to_xy(pose_landmarks[LEFT_SHOULDER])
    right_shoulder = _landmark_to_xy(pose_landmarks[RIGHT_SHOULDER])
    return (left_shoulder + right_shoulder) / 2.0


def compute_trunk_angle(pose_landmarks):
    """
    Ângulo do tronco em relação à vertical, em graus.
    0° = corpo totalmente vertical (em pé)
    90° = corpo totalmente horizontal (deitado)
    """
    hip = get_hip_center(pose_landmarks)
    shoulder = get_shoulder_center(pose_landmarks)

    trunk_vector = shoulder - hip
    vertical_vector = np.array([0.0, -1.0])

    cos_angle = np.dot(trunk_vector, vertical_vector) / (np.linalg.norm(trunk_vector) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


class TemporalFeatureExtractor:
    """
    Mantém estado entre frames para calcular features que dependem
    do frame anterior (ex: velocidade).
    """

    def __init__(self):
        self.prev_hip_y = None

    def extract(self, pose_landmarks):
        """
        Recebe os landmarks de UM frame e retorna um dicionário de features.
        Deve ser chamado uma vez por frame, em sequência (mantém estado interno).
        """
        hip_center = get_hip_center(pose_landmarks)
        trunk_angle = compute_trunk_angle(pose_landmarks)

        hip_y = hip_center[1]

        if self.prev_hip_y is None:
            vertical_velocity = 0.0
        else:
            vertical_velocity = hip_y - self.prev_hip_y

        self.prev_hip_y = hip_y

        return {
            "hip_height": 1.0 - hip_y,
            "trunk_angle_deg": trunk_angle,
            "vertical_velocity": vertical_velocity,
        }