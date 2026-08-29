import time
from enum import Enum

class FallState(Enum):
    NORMAL = "normal"
    SUSPECTED_FALL = "suspected_fall"  # pico de velocidade + inclinação detectados
    ON_GROUND = "on_ground" # pico de velocidade + inclinação + altura detectados

class FallDetector:
    """
    Detecta quedas com base em velocidade vertical, ângulo do tronco e altura do quadril.
    """
    def __init__(
        self,
        velocity_threshold: float = 0.05,      # pico de velocidade vertical p/ suspeitar
        trunk_angle_threshold: float = 45.0,    # graus, corpo "caído"
        hip_height_threshold: float = 0.4,      # abaixo disso = "no chão"
        ground_confirmation_seconds: float = 1.5,  # tempo parado no chão p/ confirmar
    ):
        self.velocity_threshold = velocity_threshold
        self.trunk_angle_threshold = trunk_angle_threshold
        self.hip_height_threshold = hip_height_threshold
        self.ground_confirmation_seconds = ground_confirmation_seconds

        self.state = FallState.NORMAL
        self.suspected_since = None
        self.alert_fired = False

    def update(self, features: dict) -> dict:
        """
        Recebe o dicionário de features de UM frame e atualiza o estado interno.
        Retorna um dicionário com o estado atual e se um alerta deve ser disparado.
        """
        vertical_velocity = features["vertical_velocity"]
        trunk_angle = features["trunk_angle_deg"]
        hip_height = features["hip_height"]

        now = time.time()
        event = None

        if self.state == FallState.NORMAL:

            # Detecta o início de uma possível queda: pico de velocidade + tronco inclinando
            if abs(vertical_velocity) > self.velocity_threshold and trunk_angle > self.trunk_angle_threshold:
                self.state = FallState.SUSPECTED_FALL
                self.suspected_since = now

        elif self.state == FallState.SUSPECTED_FALL:
            if hip_height < self.hip_height_threshold:

                # Pessoa está baixa -> começa a contar tempo no chão
                self.state = FallState.ON_GROUND
                self.suspected_since = now
            elif trunk_angle < self.trunk_angle_threshold * 0.5:

                # Tronco voltou a ficar ereto -> falso alarme (ex: agachou e levantou rápido)
                self.state = FallState.NORMAL
                self.suspected_since = None

        elif self.state == FallState.ON_GROUND:
            if hip_height < self.hip_height_threshold:
                elapsed = now - self.suspected_since
                if elapsed >= self.ground_confirmation_seconds and not self.alert_fired:
                    event = "POSSIVEL_QUEDA"
                    self.alert_fired = True
            else:
                
                # Levantou -> reseta
                self.state = FallState.NORMAL
                self.suspected_since = None
                self.alert_fired = False

        return {
            "state": self.state.value,
            "event": event,
        }