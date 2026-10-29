import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.alerts import load_events, load_reviews, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    events = load_events(args.log_dir)
    if not events:
        print("Nenhum evento registrado ainda.")
        return
    s = summarize(events, load_reviews(args.log_dir))
    h = s["hours"]

    print(f"Sessões: {s['sessions']}  (sem fim registrado: {s['unclosed_sessions']})")
    print(f"Tempo ligado: {h:.2f} h  (inclui períodos sem ninguém no quadro)")
    print(f"\nAlertas de queda do modelo: {s['fall_alerts']}")
    print(f"  revisados como queda real/simulada: {s['real']}")
    print(f"  revisados como falso alarme:        {s['false']}")
    print(f"  ainda não revisados:                {s['unreviewed']}")
    if h > 0:
        lo = s["false"] / h
        hi = (s["false"] + s["unreviewed"]) / h
        print(f"\nFalsos alarmes por hora: entre {lo:.2f} e {hi:.2f}  "
              f"(o limite superior assume que todos os não revisados eram falsos)")
    print(f"\nEpisódios que chegaram a EMERGÊNCIA: {s['emergencies']}")
    print(f"Episódios encerrados porque a pessoa se levantou: {s['resolved']}")

    if s["fps_median"] is not None:
        print(f"\nFPS (batimentos): mediana {s['fps_median']:.1f}, mínimo {s['fps_min']:.1f}, "
              f"{s['low_fps_frac']:.0%} dos batimentos abaixo de 20")
    if s["visible_median"] is not None:
        print(f"Fração do tempo com pessoa visível (mediana): {s['visible_median']:.0%}")


if __name__ == "__main__":
    main()