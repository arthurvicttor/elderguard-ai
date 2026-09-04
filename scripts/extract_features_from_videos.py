import sys
import os
import argparse
import pandas as pd
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pose import PoseEstimator
from src.features import TemporalFeatureExtractor

MIN_VISIBILITY = 0.5
RELEVANT_INDICES = [11, 12, 23, 24]  # ombros e quadris


def process_video(video_path, label, pose_estimator):
    """Processa um único vídeo e retorna uma lista de dicts (uma linha por frame válido)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [AVISO] Não foi possível abrir: {video_path}")
        return []

    feature_extractor = TemporalFeatureExtractor() 
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = pose_estimator.process(frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            avg_visibility = sum(landmarks[i].visibility for i in RELEVANT_INDICES) / len(RELEVANT_INDICES)

            if avg_visibility >= MIN_VISIBILITY:
                features = feature_extractor.extract(landmarks)
                features["frame_idx"] = frame_idx
                features["label"] = label
                features["video_file"] = os.path.basename(video_path)
                rows.append(features)

        frame_idx += 1

    cap.release()
    return rows


def process_dataset_folder(dataset_dir, output_csv):
    """
    Espera uma estrutura tipo:
        dataset_dir/
            falling/*.mp4
            walking/*.mp4
            sitting/*.mp4
            ...
    O nome da subpasta vira o label.
    """
    pose_estimator = PoseEstimator()
    all_rows = []

    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Categorias encontradas: {categories}")

    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        video_files = [f for f in os.listdir(category_path) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        print(f"\nProcessando categoria '{category}' ({len(video_files)} vídeos)...")

        for video_file in video_files:
            video_path = os.path.join(category_path, video_file)
            print(f"  -> {video_file}")
            rows = process_video(video_path, category, pose_estimator)
            all_rows.extend(rows)
            print(f"     {len(rows)} frames válidos extraídos")

    pose_estimator.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nCSV salvo em: {output_csv}")
    print(f"Total de linhas: {len(df)}")
    if not df.empty:
        print("\nDistribuição por label:")
        print(df["label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Pasta com subpastas por categoria")
    parser.add_argument("--output_csv", default="dataset/features.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    process_dataset_folder(args.dataset_dir, args.output_csv)