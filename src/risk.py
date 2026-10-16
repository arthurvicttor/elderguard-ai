from enum import Enum


class Level(Enum):
    LOW = "baixo risco"
    ATTENTION = "atencao"
    ALERT = "alerta"
    EMERGENCY = "emergencia"


# EXEMPLOS da seção 11 do plano. NÃO estão calibrados.
POINTS_FALL = 50
POINTS_NO_RECOVERY = 30   # imobilidade no chão OU pessoa fora de vista após a queda
LEVEL_THRESHOLDS = [(80, Level.EMERGENCY), (50, Level.ALERT), (30, Level.ATTENTION)]


def level_for(score):
    for limit, level in LEVEL_THRESHOLDS:
        if score >= limit:
            return level
    return Level.LOW


class RiskMonitor:
    """
    Acompanha um 'episódio' que começa em um alerta de queda do modelo e termina
    quando a pessoa se levanta (ou o operador reinicia).
    O tempo entra como argumento para a lógica poder ser testada sem câmera.
    """

    def __init__(self, immobile_s=10.0, recovery_s=3.0, lost_s=5.0):
        self.immobile_s = immobile_s   # parado no chão por este tempo -> +30
        self.recovery_s = recovery_s   # de pé por este tempo -> episódio encerrado
        self.lost_s = lost_s           # pose perdida por este tempo -> +30
        self.reset()

    def reset(self):
        self.active = False
        self.still_since = None
        self.upright_since = None
        self.lost_since = None

    def _result(self, score, reasons, resolved=False):
        return {"score": score, "level": level_for(score), "reasons": reasons,
                "active": self.active, "resolved": resolved}

    def update(self, now, fall_alert=False, pose_ok=True,
               on_ground_still=False, upright=False):
        if fall_alert and not self.active:
            self.reset()
            self.active = True
        if not self.active:
            return self._result(0, [])

        if not pose_ok:
            # Não zera still_since: uma oscilação curta do rastreamento não deve
            # impedir o alerta. Só passa a contar o tempo fora de vista.
            if self.lost_since is None:
                self.lost_since = now
            self.upright_since = None
        else:
            self.lost_since = None
            if upright:
                if self.upright_since is None:
                    self.upright_since = now
                self.still_since = None
                if now - self.upright_since >= self.recovery_s:
                    self.reset()
                    return self._result(0, ["pessoa se levantou"], resolved=True)
            else:
                self.upright_since = None
                if on_ground_still:
                    if self.still_since is None:
                        self.still_since = now
                else:
                    self.still_since = None

        score, reasons = POINTS_FALL, ["possivel queda"]
        if self.still_since is not None and now - self.still_since >= self.immobile_s:
            score += POINTS_NO_RECOVERY
            reasons.append(f"imovel no chao ha {now - self.still_since:.0f}s")
        elif self.lost_since is not None and now - self.lost_since >= self.lost_s:
            score += POINTS_NO_RECOVERY
            reasons.append("pessoa fora de vista apos queda")
        return self._result(score, reasons)