import glob
import json
import os
import statistics
from datetime import datetime

DEFAULT_LOG_DIR = "logs"


class EventLog:
    """
    Log local em JSON Lines, um arquivo por dia.
    Guarda só números e texto: nenhum vídeo ou imagem é salvo.
    """

    def __init__(self, log_dir=DEFAULT_LOG_DIR, enabled=True):
        self.log_dir = log_dir
        self.enabled = enabled
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._n = 0
        if enabled:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, event_type, **data):
        if not self.enabled:
            return None
        self._n += 1
        now = datetime.now()
        record = {
            "id": f"{self.session_id}-{self._n:04d}",
            "ts": now.isoformat(timespec="seconds"),
            "session": self.session_id,
            "type": event_type,
            **data,
        }
        path = os.path.join(self.log_dir, f"events_{now:%Y-%m-%d}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def _read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # linha cortada por um travamento
    return out


def load_events(log_dir=DEFAULT_LOG_DIR):
    events = []
    for path in sorted(glob.glob(os.path.join(log_dir, "events_*.jsonl"))):
        events.extend(_read_jsonl(path))
    return events


def load_reviews(log_dir=DEFAULT_LOG_DIR):
    """Retorna {id_do_evento: 'real' | 'false'}. A revisão mais recente vence."""
    reviews = {}
    for r in _read_jsonl(os.path.join(log_dir, "reviews.jsonl")):
        reviews[r["id"]] = r["verdict"]
    return reviews


def save_review(event_id, verdict, log_dir=DEFAULT_LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    rec = {"id": event_id, "verdict": verdict,
           "reviewed_at": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(log_dir, "reviews.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize(events, reviews=None):
    reviews = reviews or {}

    by_session = {}
    for e in events:
        by_session.setdefault(e["session"], []).append(e)

    total_s, unclosed, lost_s = 0.0, 0, 0.0
    best_mem = []
    for evs in by_session.values():
        evs = sorted(evs, key=lambda e: e["ts"])
        start_ev = next((e for e in evs if e["type"] == "session_start"), evs[0])
        end_ev = next((e for e in evs if e["type"] == "session_end"), None)
        if end_ev is None:
            unclosed += 1
            end_ev = evs[-1]          # sem fim: usa o último evento registrado
        start = datetime.fromisoformat(start_ev["ts"])
        end = datetime.fromisoformat(end_ev["ts"])
        span = max((end - start).total_seconds(), 0.0)
        # tempo em que o sistema não estava vendo nada (PC suspenso, câmera caída)
        down = min(sum(e.get("seconds", 0.0) for e in evs
                       if e["type"] in ("time_gap", "camera_recovered")), span)
        lost_s += down
        total_s += span - down

        mem = [e["rss_mb"] for e in evs if e["type"] == "heartbeat" and "rss_mb" in e]
        if len(mem) > len(best_mem):
            best_mem = mem            # sessão mais longa com medida de memória
    hours = total_s / 3600.0

    alerts = [e for e in events if e["type"] == "fall_alert"]
    verdicts = [reviews.get(a["id"]) for a in alerts]
    real = verdicts.count("real")
    false = verdicts.count("false")
    unreviewed = len(alerts) - real - false

    changes = [e for e in events if e["type"] == "risk_change"]
    beats = [e for e in events if e["type"] == "heartbeat"]
    fps = [e["fps"] for e in beats if "fps" in e]
    vis = [e["person_visible"] for e in beats if "person_visible" in e]
    cpu = [e["cpu_pct"] for e in beats if "cpu_pct" in e]

    return {
        "sessions": len(by_session),
        "unclosed_sessions": unclosed,
        "hours": hours,
        "lost_s": lost_s,
        "camera_losses": sum(1 for e in events if e["type"] == "camera_lost"),
        "time_gaps": sum(1 for e in events if e["type"] == "time_gap"),
        "crashes": sum(1 for e in events if e["type"] == "crash"),
        "fall_alerts": len(alerts),
        "real": real, "false": false, "unreviewed": unreviewed,
        "emergencies": sum(1 for c in changes if c.get("to_level") == "emergencia"),
        "resolved": sum(1 for c in changes if "pessoa se levantou" in c.get("reasons", [])),
        "fps_median": statistics.median(fps) if fps else None,
        "fps_min": min(fps) if fps else None,
        "low_fps_frac": (sum(1 for x in fps if x < 20) / len(fps)) if fps else None,
        "visible_median": statistics.median(vis) if vis else None,
        "cpu_median": statistics.median(cpu) if cpu else None,
        "mem_first_mb": best_mem[0] if len(best_mem) >= 2 else None,
        "mem_last_mb": best_mem[-1] if len(best_mem) >= 2 else None,
        "mem_beats": len(best_mem),
    }