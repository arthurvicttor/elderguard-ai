import os
import argparse
import cv2

HELP1 = "d/a: +-1 frame   w/x: +-10 frames"
HELP2 = "b: INICIO   e: FIM   k: salvar   n: pular   q: sair"


def load_frames(path, max_w=640):
    """Carrega o vídeo inteiro, em ordem, para o índice bater com a extração."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > max_w:
            s = max_w / f.shape[1]
            f = cv2.resize(f, None, fx=s, fy=s)
        frames.append(f)
    cap.release()
    return frames


def annotate(frames, key):
    idx, start, end = 0, None, None
    n = len(frames)
    while True:
        view = frames[idx].copy()
        lines = [f"{key}  frame {idx + 1}/{n}", f"inicio={start}  fim={end}", HELP1, HELP2]
        for i, t in enumerate(lines):
            cv2.putText(view, t, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
        cv2.imshow("Anotar quedas", view)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("d"):
            idx = min(idx + 1, n - 1)
        elif k == ord("a"):
            idx = max(idx - 1, 0)
        elif k == ord("w"):
            idx = min(idx + 10, n - 1)
        elif k == ord("x"):
            idx = max(idx - 10, 0)
        elif k == ord("b"):
            start = idx + 1          # numeração a partir de 1, como no Le2i
        elif k == ord("e"):
            end = idx + 1
        elif k == ord("k"):
            if start is not None and end is not None and end >= start:
                return "save", start, end
            print("Marque o início (b) e o fim (e), com fim >= início.")
        elif k == ord("n"):
            return "skip", None, None
        elif k == ord("q"):
            return "quit", None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="dataset/falling")
    parser.add_argument("--output", default="dataset/own_annotations.csv")
    args = parser.parse_args()

    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            done = {line.split(",")[0] for line in f.readlines()[1:]}

    folder = os.path.basename(os.path.normpath(args.dir))
    videos = sorted(v for v in os.listdir(args.dir) if v.lower().endswith((".mp4", ".avi", ".mov")))

    for v in videos:
        key = f"{folder}/{v}"
        if key in done:
            continue
        frames = load_frames(os.path.join(args.dir, v))
        if not frames:
            print(f"[AVISO] vídeo vazio: {key}")
            continue
        action, s, e = annotate(frames, key)
        if action == "quit":
            break
        if action == "save":
            new = not os.path.exists(args.output)
            with open(args.output, "a", encoding="utf-8") as f:
                if new:
                    f.write("video,fall_start,fall_end\n")
                f.write(f"{key},{s},{e}\n")
            print(f"Salvo: {key} -> {s}-{e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()