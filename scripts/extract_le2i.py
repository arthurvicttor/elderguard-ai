import sys
import os
import argparse
import cv2
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.pose import PoseEstimator
from src.features import TemporalFeatureExtractor

MIN_VISIBILITY = 0.5
CORE = [11, 12, 23, 24]  # ombros e quadris

META = ["scene", "video_file", "frame_idx", "frame_number",
        "phase", "is_fall", "pose_ok", "mean_visibility"]
LANDMARKS = [f"lm{i}_{c}" for i in CORE for c in ("x", "y", "vis")]
FEATURES = ["hip_height", "trunk_angle_deg", "vertical_velocity"]
COLUMNS = META + LANDMARKS + FEATURES


def find_scenes(root):
    """Acha pastas que tenham 'Videos' e 'Annotation_files' lado a lado."""
    scenes = []
    for dirpath, dirnames, _ in os.walk(root):
        if "Videos" in dirnames and "Annotation_files" in dirnames:
            scenes.append(dirpath)
    return sorted(scenes)


def read_fall_interval(txt_path):
    """Retorna (inicio, fim) da queda, ou None se o vídeo não tiver queda."""
    if not os.path.exists(txt_path):
        return None
    with open(txt_path, encoding="utf-8-sig", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines()[:2]]
    try:
        start, end = int(lines[0]), int(lines[1])
    except (ValueError, IndexError):
        return None
    if start <= 0 or end < start:
        return None
    return start, end


def process_video(video_path, scene, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [AVISO] Não foi possível abrir: {video_path}")
        return []

    pose = PoseEstimator()
    extractor = TemporalFeatureExtractor()
    video_file = os.path.basename(video_path)
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number = frame_idx + 1  # as anotações do Le2i começam em 1

        if interval is None:
            phase, is_fall = "adl", 0
        elif frame_number < interval[0]:
            phase, is_fall = "pre", 0
        elif frame_number <= interval[1]:
            phase, is_fall = "fall", 1
        else:
            phase, is_fall = "post", 0

        row = {
            "scene": scene, "video_file": video_file,
            "frame_idx": frame_idx, "frame_number": frame_number,
            "phase": phase, "is_fall": is_fall, "pose_ok": 0,
        }

        result = pose.process(frame)
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            vis = sum(lms[i].visibility for i in CORE) / len(CORE)
            row["mean_visibility"] = float(vis)
            for i in CORE:
                row[f"lm{i}_x"] = float(lms[i].x)
                row[f"lm{i}_y"] = float(lms[i].y)
                row[f"lm{i}_vis"] = float(lms[i].visibility)
            if vis >= MIN_VISIBILITY:
                row.update(extractor.extract(lms))
                row["pose_ok"] = 1

        rows.append(row)
        frame_idx += 1

    cap.release()
    pose.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset_raw")
    parser.add_argument("--output", default="dataset/features_le2i.csv")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Limita vídeos por cena (útil para testar)")
    args = parser.parse_args()

    scenes = find_scenes(args.root)
    if not scenes:
        print("Nenhuma pasta com 'Videos' + 'Annotation_files' encontrada.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        os.remove(args.output)

    for scene_dir in scenes:
        scene = os.path.basename(scene_dir)
        videos_dir = os.path.join(scene_dir, "Videos")
        ann_dir = os.path.join(scene_dir, "Annotation_files")
        video_files = sorted(f for f in os.listdir(videos_dir)
                             if f.lower().endswith((".avi", ".mp4")))
        if args.max_videos:
            video_files = video_files[:args.max_videos]

        print(f"\n[{scene}] {len(video_files)} vídeos")
        for vf in video_files:
            stem = os.path.splitext(vf)[0]
            interval = read_fall_interval(os.path.join(ann_dir, stem + ".txt"))
            rows = process_video(os.path.join(videos_dir, vf), scene, interval)
            if not rows:
                continue

            ok_rate = sum(r["pose_ok"] for r in rows) / len(rows)
            tag = f"queda {interval[0]}-{interval[1]}" if interval else "sem queda"
            print(f"  {vf}: {len(rows)} frames, pose ok {ok_rate:.0%}, {tag}")

            df = pd.DataFrame(rows).reindex(columns=COLUMNS)
            df.to_csv(args.output, mode="a",
                      header=not os.path.exists(args.output), index=False)

    if not os.path.exists(args.output):
        return
    df = pd.read_csv(args.output)
    print("\nFrames por fase e taxa de pose válida:")
    print(df.groupby("phase")["pose_ok"].agg(["count", "mean"]).round(3))
    per_video = df.groupby(["scene", "video_file"])["is_fall"].max()
    print(f"\nVídeos com queda: {int(per_video.sum())} | "
          f"sem queda: {int((per_video == 0).sum())}")
    print(f"CSV salvo em: {args.output}")


if __name__ == "__main__":
    main()