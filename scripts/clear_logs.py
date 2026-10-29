import glob
import os
import argparse
from datetime import datetime, timedelta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--older-than", type=int, default=None,
                   help="Apaga só arquivos de eventos com mais de N dias (mantém as revisões)")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "events_*.jsonl")))
    if args.older_than is not None:
        cutoff = datetime.now() - timedelta(days=args.older_than)
        files = [f for f in files
                 if datetime.strptime(os.path.basename(f)[7:17], "%Y-%m-%d") < cutoff]
    else:
        reviews = os.path.join(args.log_dir, "reviews.jsonl")
        if os.path.exists(reviews):
            files.append(reviews)

    if not files:
        print("Nada para apagar.")
        return
    print("Serão apagados:")
    for f in files:
        print("  ", f)
    if input("Confirma? [s/N]: ").strip().lower() != "s":
        print("Cancelado.")
        return
    for f in files:
        os.remove(f)
    print(f"{len(files)} arquivo(s) apagado(s).")


if __name__ == "__main__":
    main()