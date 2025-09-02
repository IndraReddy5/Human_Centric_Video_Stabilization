"""Detects and stores the human pose landmarks and also identifies the anchor point for stabilization."""

import mediapipe as mp


class PoseDetector:
    """Detects human poses in video frames."""

    def __init__(self):
        """Initializes the pose detector."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    def find_anchor_point(self, frame_rgb):
        """Finds the anchor point in the frame."""
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None, None

        landmarks = results.pose_landmarks.landmark
        h, w, _ = frame_rgb.shape

        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # selecting hip level or shoulder level points as the anchor point
        if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
            anchor_x = int((left_hip.x + right_hip.x) * w / 2)
            anchor_y = int((left_hip.y + right_hip.y) * h / 2)
        elif left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            anchor_x = int((left_shoulder.x + right_shoulder.x) * w / 2)
            anchor_y = int((left_shoulder.y + right_shoulder.y) * h / 2)
        else:
            return None, None

        all_landmarks_data = {
            "frame_landmarks": [
                {
                    "name": name,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
                for name, lm in zip(self.mp_pose.PoseLandmark, landmarks)
            ]
        }

        return (anchor_x, anchor_y), all_landmarks_data

    def close(self):
        """Closes the pose detector."""
        self.pose.close()
