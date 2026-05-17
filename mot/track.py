import numpy as np

from mot.config import initial_covariance, model_matrices
from mot.kalman import KalmanFilter


class Track:
    def __init__(self, measurement, track_id, config, status="candidate"):
        F, H, Q, R = model_matrices(config)
        x, y, w, h = np.asarray(measurement, dtype=float)
        state = np.array([x, 0.0, y, 0.0, w, h])
        self.kf = KalmanFilter(state, initial_covariance(config), F, H, Q, R)
        self.id = track_id
        self.status = status
        self.hits = []
        self.history = []

    def predict(self):
        self.kf.predict()

    def update(self, measurement, time):
        self.kf.update(measurement)
        self.hits.append(1)
        self.record(time)

    def update_jpda(self, measurements, probabilities, time):
        self.kf.update_jpda(measurements, probabilities)
        self.hits.append(1 if np.sum(probabilities) > 0.5 else 0)
        self.record(time)

    def miss(self, time):
        self.hits.append(0)
        self.record(time)

    def record(self, time):
        row = np.r_[time, self.id, self.kf.state, self.kf.predicted_measurement]
        self.history.append(row)

    def recent_hits(self, n):
        return self.hits[-n:]

    def as_array(self):
        if not self.history:
            return np.empty((0, 12))
        return np.asarray(self.history, dtype=float)


class TrackManager:
    def __init__(self, config):
        self.config = config
        self.next_id = 0
        self.candidate_tracks = []
        self.confirmed_tracks = []
        self.deleted_tracks = []

    def all_active(self):
        return self.confirmed_tracks + self.candidate_tracks

    def create_tracks(self, measurements):
        for measurement in measurements:
            self.candidate_tracks.append(Track(measurement, self.next_id, self.config))
            self.next_id += 1

    def step_lifecycle(self):
        self._confirm_candidates()
        self._delete_stale(self.confirmed_tracks, keep_deleted=True)
        self._delete_stale(self.candidate_tracks, keep_deleted=False)

    def _confirm_candidates(self):
        kept = []
        for track in self.candidate_tracks:
            window = track.recent_hits(self.config["confirm_n"])
            if window.count(1) >= self.config["confirm_m"]:
                track.status = "confirmed"
                self.confirmed_tracks.append(track)
            else:
                kept.append(track)
        self.candidate_tracks = kept

    def _delete_stale(self, tracks, keep_deleted):
        kept = []
        for track in tracks:
            window = track.recent_hits(self.config["delete_n"])
            if len(window) >= self.config["delete_n"] and window.count(0) >= self.config["delete_m"]:
                track.status = "deleted"
                if keep_deleted:
                    self.deleted_tracks.append(track)
            else:
                kept.append(track)
        tracks[:] = kept

    def result(self, tracker_name, config):
        tracks = self.deleted_tracks + self.confirmed_tracks + self.candidate_tracks
        histories = np.empty(len(tracks), dtype=object)
        for i, track in enumerate(tracks):
            histories[i] = track.as_array()
        return {
            "tracker": tracker_name,
            "tracks": histories,
            "track_ids": np.array([track.id for track in tracks], dtype=int),
            "config": dict(config),
        }
