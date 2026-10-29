import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.alerts import load_events, load_reviews, save_review


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="logs")
    args = p.parse_args()

    events = load_events(args.log_dir)
    reviews = load_reviews(args.log_dir)
    pending = [e for e in events if e["type"] == "fall_alert" and e["id"] not in reviews]
    print(f"{len(pending)} alertas para revisar.")
    print("Como não gravamos vídeo, você precisa lembrar o que estava acontecendo naquele horário.")

    for e in pending:
        print(f"\n{e['ts']}  score {e.get('score')}  (id {e['id']})")
        ans = input("[r] queda real ou simulada por mim  [f] falso alarme  [p] pular  [q] sair: ").strip().lower()
        if ans == "q":
            break
        if ans == "r":
            save_review(e["id"], "real", args.log_dir)
        elif ans == "f":
            save_review(e["id"], "false", args.log_dir)


if __name__ == "__main__":
    main()