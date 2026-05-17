import numpy as np


class KalmanFilter:
    def __init__(self, state, covariance, F, H, Q, R):
        self.state = np.array(state, dtype=float)
        self.covariance = np.array(covariance, dtype=float)
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.predicted_measurement = self.H @ self.state
        self.innovation_covariance = self.H @ self.covariance @ self.H.T + self.R

    def predict(self):
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        self.predicted_measurement = self.H @ self.state
        self.innovation_covariance = self.H @ self.covariance @ self.H.T + self.R

    def update(self, measurement):
        measurement = np.asarray(measurement, dtype=float)
        S = self.innovation_covariance
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.predicted_measurement
        self.state = self.state + K @ innovation
        I = np.eye(self.covariance.shape[0])
        self.covariance = (I - K @ self.H) @ self.covariance
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.predicted_measurement = self.H @ self.state

    def update_jpda(self, measurements, probabilities):
        measurements = np.asarray(measurements, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        miss_probability = max(0.0, 1.0 - probabilities.sum())

        S = self.innovation_covariance
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        P_update = self.covariance - K @ S @ K.T

        if len(measurements) == 0 or probabilities.sum() == 0:
            return

        innovations = measurements - self.predicted_measurement
        weighted_innovation = probabilities @ innovations
        self.state = self.state + K @ weighted_innovation

        spread = np.zeros_like(self.covariance)
        for prob, innovation in zip(probabilities, innovations):
            residual = K @ innovation - K @ weighted_innovation
            spread += prob * np.outer(residual, residual)

        self.covariance = probabilities.sum() * P_update + miss_probability * self.covariance + spread
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.predicted_measurement = self.H @ self.state

    def mahalanobis2(self, measurement):
        innovation = np.asarray(measurement, dtype=float) - self.predicted_measurement
        return float(innovation.T @ np.linalg.inv(self.innovation_covariance) @ innovation)

