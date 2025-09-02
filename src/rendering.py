"""Handles video rendering and saving."""

import json
import os
import cv2
import numpy as np


def create_video_writers(output_dir, w, h, fps):
    """Creates video writers for stabilized and comparison videos."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stabilized_writer = cv2.VideoWriter(
        f"{output_dir}/stabilized_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    comparison_writer = cv2.VideoWriter(
        f"{output_dir}/comparison_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h),
    )
    return stabilized_writer, comparison_writer


def write_comparison_frame(original_frame, stabilized_frame, writer):
    """Writes a side-by-side comparison frame to the writer."""
    cv2.putText(
        original_frame,
        "Original",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        stabilized_frame,
        "Stabilized",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    comparison_frame = np.hstack((original_frame, stabilized_frame))
    writer.write(comparison_frame)


def save_pose_data(output_dir, pose_data):
    """Saves the pose data to a JSON file."""
    output_path = f"{output_dir}/pose_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, indent=4)
