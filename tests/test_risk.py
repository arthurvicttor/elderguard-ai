import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.risk import RiskMonitor, Level  # pyright: ignore[reportMissingImports]

GROUND = dict(pose_ok=True, on_ground_still=True, upright=False)


def run(m, t0, t1, step=0.5, **kw):
    out, t = None, t0
    while t <= t1:
        out = m.update(t, **kw)
        t += step
    return out


def test_sem_queda_fica_baixo():
    m = RiskMonitor()
    r = run(m, 0, 30, upright=True)
    assert r["level"] == Level.LOW and r["score"] == 0


def test_queda_gera_alerta_e_imobilidade_gera_emergencia():
    m = RiskMonitor(immobile_s=10)
    r = m.update(0, fall_alert=True, **GROUND)
    assert r["score"] == 50 and r["level"] == Level.ALERT
    r = run(m, 0.5, 9.5, **GROUND)
    assert r["score"] == 50
    r = m.update(10, **GROUND)
    assert r["score"] == 80 and r["level"] == Level.EMERGENCY


def test_levantar_resolve_o_episodio():
    m = RiskMonitor(recovery_s=3)
    m.update(0, fall_alert=True, **GROUND)
    r = run(m, 1, 4, pose_ok=True, upright=True)
    assert r["resolved"] and r["level"] == Level.LOW
    r = m.update(5, **GROUND)          # sem novo alerta de queda, nada acontece
    assert r["score"] == 0


def test_pessoa_fora_de_vista_apos_queda():
    m = RiskMonitor(lost_s=5)
    m.update(0, fall_alert=True, pose_ok=True)
    r = run(m, 1, 5.5, pose_ok=False)
    assert r["score"] == 50
    r = m.update(6, pose_ok=False)
    assert r["score"] == 80 and any("fora de vista" in x for x in r["reasons"])


def test_oscilacao_curta_nao_zera_a_imobilidade():
    m = RiskMonitor(immobile_s=10)
    m.update(0, fall_alert=True, **GROUND)
    run(m, 0.5, 3, **GROUND)
    m.update(3.5, pose_ok=False)
    r = run(m, 4, 10, **GROUND)
    assert r["score"] == 80


def test_segundo_alerta_nao_reinicia_o_episodio():
    m = RiskMonitor(immobile_s=10)
    m.update(0, fall_alert=True, **GROUND)
    run(m, 0.5, 4.5, **GROUND)
    m.update(5, fall_alert=True, **GROUND)
    r = m.update(10, **GROUND)
    assert r["score"] == 80


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("Todos os testes passaram.")