import os
import sys
import time
import argparse
import subprocess


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--max-restarts", type=int, default=10)
    p.add_argument("--backoff-s", type=float, default=10.0)
    args, extra = p.parse_known_args()        # o resto vai direto para o live_detect.py

    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "live_detect.py"), "--headless"] + extra

    restarts = 0
    try:
        while True:
            print(f"[supervisor] iniciando (reinicios ate agora: {restarts})")
            proc = subprocess.Popen(cmd)
            try:
                code = proc.wait()
            except KeyboardInterrupt:
                print("[supervisor] Ctrl+C: aguardando o detector encerrar...")
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                return
            if code == 0:
                print("[supervisor] encerrado normalmente.")
                return
            restarts += 1
            if restarts > args.max_restarts:
                print("[supervisor] reinicios demais, desistindo.")
                return
            print(f"[supervisor] saiu com codigo {code}; reiniciando em {args.backoff_s:.0f}s")
            time.sleep(args.backoff_s)
    except KeyboardInterrupt:
        print("[supervisor] interrompido.")


if __name__ == "__main__":
    main()