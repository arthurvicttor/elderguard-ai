import sys
import os
import argparse
import time
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.camera import Camera


def record_clip(output_path, duration_seconds=6, fps=20):
    camera = Camera(source=0).open()
    frame_width = int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Gravando por {duration_seconds}s... Prepare-se!")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    print("GRAVANDO!")

    start = time.time()
    try:
        while time.time() - start < duration_seconds:
            frame = camera.read_frame()
            if frame is None:
                break
            writer.write(frame)
            cv2.imshow("Gravando... (ESC para parar antes)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="Ex: falling, walking, sitting")
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--duration", type=int, default=6)
    args = parser.parse_args()

    category_dir = os.path.join(args.dataset_dir, args.category)
    os.makedirs(category_dir, exist_ok=True)

    existing = [f for f in os.listdir(category_dir) if f.endswith(".mp4")]
    next_idx = len(existing) + 1
    output_path = os.path.join(category_dir, f"{args.category}_{next_idx:03d}.mp4")

    record_clip(output_path, duration_seconds=args.duration)