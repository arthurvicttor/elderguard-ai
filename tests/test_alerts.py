import sys
import os
import tempfile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.alerts import EventLog, load_events, summarize


def ev(id_, ts, type_, session="s1", **kw):
    return {"id": id_, "ts": ts, "session": session, "type": type_, **kw}


def test_log_escreve_uma_linha_por_evento():
    with tempfile.TemporaryDirectory() as d:
        log = EventLog(d)
        log.log("fall_alert", score=0.9)
        log.log("heartbeat", fps=25.0)
        files = [f for f in os.listdir(d) if f.startswith("events_")]
        assert len(files) == 1
        events = load_events(d)
        assert [e["type"] for e in events] == ["fall_alert", "heartbeat"]
        assert events[0]["score"] == 0.9 and "id" in events[0] and "ts" in events[0]


def test_log_desligado_nao_cria_nada():
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "logs")
        EventLog(sub, enabled=False).log("fall_alert", score=0.9)
        assert not os.path.exists(sub)


def test_linha_corrompida_e_ignorada():
    with tempfile.TemporaryDirectory() as d:
        EventLog(d).log("heartbeat", fps=25.0)
        path = os.path.join(d, os.listdir(d)[0])
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"id": "quebrada...\n')
        assert len(load_events(d)) == 1


def test_resumo_horas_e_alarmes():
    events = [
        ev("a1", "2026-10-30T10:00:00", "session_start"),
        ev("a2", "2026-10-30T10:30:00", "heartbeat", fps=25.0, person_visible=0.9),
        ev("a3", "2026-10-30T10:40:00", "fall_alert", score=0.9),
        ev("a4", "2026-10-30T10:50:00", "fall_alert", score=0.85),
        ev("a5", "2026-10-30T11:00:00", "session_end"),
    ]
    s = summarize(events, {"a3": "false"})
    assert abs(s["hours"] - 1.0) < 1e-6
    assert s["fall_alerts"] == 2 and s["false"] == 1 and s["unreviewed"] == 1
    assert s["unclosed_sessions"] == 0


def test_sessao_sem_fim_usa_ultimo_evento():
    events = [
        ev("b1", "2026-10-30T10:00:00", "session_start", session="s2"),
        ev("b2", "2026-10-30T10:30:00", "heartbeat", session="s2", fps=25.0),
    ]
    s = summarize(events)
    assert abs(s["hours"] - 0.5) < 1e-6 and s["unclosed_sessions"] == 1

def test_resumo_desconta_tempo_perdido():
    events = [
        ev("c1", "2026-10-30T10:00:00", "session_start", session="s3"),
        ev("c2", "2026-10-30T10:20:00", "time_gap", session="s3", seconds=600.0),
        ev("c3", "2026-10-30T10:40:00", "camera_recovered", session="s3", seconds=300.0),
        ev("c4", "2026-10-30T11:00:00", "session_end", session="s3"),
    ]
    s = summarize(events)
    # 3600 s de intervalo - 900 s perdidos = 2700 s = 0.75 h
    assert abs(s["hours"] - 0.75) < 1e-6 and abs(s["lost_s"] - 900.0) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("Todos os testes passaram.")