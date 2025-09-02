"""Handles video stabilization using Kalman Filter."""

import numpy as np


class KalmanFilter:
    """Kalman Filter for 2D point tracking."""

    def __init__(self, dt=1, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1):
        """Initializes the Kalman filter."""
        self.a = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.b = np.array([[(dt**2) / 2, 0], [dt, 0], [0, (dt**2) / 2], [0, dt]])
        self.h = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.q = (
            np.array(
                [
                    [(dt**4) / 4, (dt**3) / 2, 0, 0],
                    [(dt**3) / 2, dt**2, 0, 0],
                    [0, 0, (dt**4) / 4, (dt**3) / 2],
                    [0, 0, (dt**3) / 2, dt**2],
                ]
            )
            * std_acc**2
        )
        self.r = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])
        self.p = np.eye(self.a.shape[1])
        self.x = np.zeros((self.a.shape[1], 1))
        self.u = np.array([[u_x], [u_y]])

    def predict(self):
        """Predicts the next state."""
        self.x = np.dot(self.a, self.x) + np.dot(self.b, self.u)
        self.p = np.dot(np.dot(self.a, self.p), self.a.T) + self.q
        return self.x

    def update(self, z):
        """Updates the state with the new measurement."""
        s = np.dot(self.h, np.dot(self.p, self.h.T)) + self.r
        k = np.dot(np.dot(self.p, self.h.T), np.linalg.inv(s))
        y = z - np.dot(self.h, self.x)
        self.x = self.x + np.dot(k, y)
        i = np.eye(self.h.shape[1])
        self.p = np.dot(i - np.dot(k, self.h), self.p)
        return self.x


def smooth_trajectory(raw_points):
    """Smooths the trajectory using a Kalman filter."""
    kf = KalmanFilter()
    smoothed_points = []
    initial_state = np.array([[raw_points[0][0]], [0], [raw_points[0][1]], [0]])
    kf.x = initial_state
    for point in raw_points:
        measurement = np.array([[point[0]], [point[1]]])
        kf.predict()
        smoothed_state = kf.update(measurement)
        smoothed_points.append((int(smoothed_state[0, 0]), int(smoothed_state[2, 0])))
    return np.array(smoothed_points)
