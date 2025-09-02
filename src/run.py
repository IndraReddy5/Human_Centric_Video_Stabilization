"""Main entry point for the human-centric stabilization pipeline."""

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

# from background import BackgroundRemover
from pose import PoseDetector
from stabilization import smooth_trajectory
from rendering import create_video_writers, write_comparison_frame, save_pose_data


def transform_point(point, transform_matrix):
    """Applies a transformation matrix to a 2D point."""
    x, y = point
    a, b, c = transform_matrix[0]
    d, e, f = transform_matrix[1]
    x_new = a * x + b * y + c
    y_new = d * x + e * y + f
    return (int(x_new), int(y_new))


def main(args):
    """Runs the main stabilization pipeline."""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input_video}")
        return

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Total frames in the video: {n_frames}")
    print(f"Video resolution: {w}x{h}, FPS: {fps}")

    pose_detector = PoseDetector()
    # bg_remover = BackgroundRemover(device=args.device)

    print("Pass 1/3: Detecting pose and collecting trajectory data...")
    raw_anchor_points = []
    all_pose_data = []

    for i in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        anchor_point, pose_data = pose_detector.find_anchor_point(frame_rgb)

        # if we are unable to find anchor points, using previous point as anchor point
        if anchor_point is None:
            if len(raw_anchor_points) > 0:
                raw_anchor_points.append(raw_anchor_points[-1])
            else:
                raw_anchor_points.append((w // 2, h // 2))
        else:
            raw_anchor_points.append(anchor_point)
            all_pose_data.append({"frame": i, **pose_data})

    cap.release()

    save_pose_data(args.output_dir, all_pose_data)
    print(f"Pose data saved to {args.output_dir}/pose_data.json")

    print("Pass 2/3: Smoothing trajectory with Kalman Filter...")
    raw_anchor_points = np.array(raw_anchor_points)
    smoothed_anchor_points = smooth_trajectory(raw_anchor_points)

    target_point = (w // 2, h // 2)

    transforms = []
    for i in range(n_frames):
        dx = target_point[0] - smoothed_anchor_points[i][0]
        dy = target_point[1] - smoothed_anchor_points[i][1]
        t = np.zeros((2, 3), dtype=np.float32)
        t[0, 0] = 1
        t[1, 1] = 1
        t[0, 2] = dx
        t[1, 2] = dy
        transforms.append(t)

    print("Pass 3/3: Applying transforms and rendering videos...")
    cap = cv2.VideoCapture(args.input_video)
    stabilized_writer, comparison_writer = create_video_writers(
        args.output_dir, w, h, fps
    )

    for i in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # foreground_frame = bg_remover.remove_background(frame)
        # stabilized_frame = cv2.warpAffine(foreground_frame, transforms[i], (w, h))
        stabilized_frame = cv2.warpAffine(frame, transforms[i], (w, h))
        tar_x, tar_y = transform_point(
            (smoothed_anchor_points[i][0], smoothed_anchor_points[i][1]), transforms[i]
        )
        cv2.circle(stabilized_frame, (tar_x, tar_y), 15, (0, 255, 0), -1)
        cv2.circle(
            frame,
            (smoothed_anchor_points[i][0], smoothed_anchor_points[i][1]),
            15,
            (0, 255, 0),
            -1,
        )
        cv2.circle(frame, target_point, 15, (255, 0, 0), -1)
        stabilized_writer.write(stabilized_frame)
        write_comparison_frame(frame, stabilized_frame, comparison_writer)

    cap.release()
    pose_detector.close()
    stabilized_writer.release()
    comparison_writer.release()

    print("Processing complete!")
    print(f"Stabilized video saved to: {args.output_dir}/stabilized_video.mp4")
    print(f"Comparison video saved to: {args.output_dir}/comparison_video.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human-Centric Video Stabilization Pipeline"
    )
    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to the input video file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run DeepLabv3 on ('cpu' or 'cuda').",
    )

    args = parser.parse_args()
    main(args)
