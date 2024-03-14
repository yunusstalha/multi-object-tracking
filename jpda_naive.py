
import numpy as np
from track_holder import TrackHolder
from scipy.spatial.distance import mahalanobis
from scipy.stats.distributions import chi2
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from track_object import TrackObject

P_DET = 0.95
P_GATE = 0.9997
BETA_FA = 10 / (1000 ** 2) 
BETA_NT = 3 / 350/(1000 ** 2)
GATE_THRESHOLD = np.sqrt(chi2.ppf(P_GATE, df=4)) 
time = 0
all_measurements = np.load("/home/talha/git/multi-object-tracking/measurements.npy", allow_pickle=True)

def check_distance(kf_obj, meas):
    return mahalanobis(meas, kf_obj.get_output(), np.linalg.inv(kf_obj.get_innovation_covariance()))

def MakeTimeUpdate(tracks):
    for track in tracks:
        track.predict()
    return tracks


def InitializeTracks(tracks, measurements):
    """
    Initialize track objects for each measurement and append them to the tracks list.

    :param tracks: List of existing track objects.
    :param measurements: List of measurements, each a tuple or list of values.
    :return: Updated list of track objects including those initialized from measurements.
    """
    for measurement in measurements:
        # Directly construct the mean array and create a new TrackObject
        mean = np.array([measurement[0], 0, measurement[1], 0, measurement[2], measurement[3]])
        track = TrackObject(mean)
        tracks.append(track)

    return tracks

def create_validation_matrix(tracks, measurements, gate_threshold):
    validation_matrix = np.zeros((len(measurements), len(tracks)))
    for i, measurement in enumerate(measurements):
        for j, track in enumerate(tracks):
            if check_distance(track, measurement) <= gate_threshold:
                validation_matrix[i, j] = 1
    return validation_matrix

def calculate_track_associations(tracks, measurements, validation_matrix):
    """
    Finds possible measurement associations for each track.

    Args:
        tracks: A list of TrackObject instances.
        measurements: A NumPy array of measurements.
        validation_matrix: A NumPy array indicating valid track-measurement associations.

    Returns:
        A list where each element is a list of possible measurement indices
        (including 0 for no association) for the corresponding track.
    """
    tracks_possible_associations = []
    for i, track in enumerate(tracks):
        valid_measurement_indices = np.nonzero(validation_matrix[:, i])[0]  # Find indices with True values
        associations = [0] + (valid_measurement_indices + 1).tolist()  # Add 1 for measurement indexing
        tracks_possible_associations.append(associations)

    return tracks_possible_associations

import itertools

def generate_hypotheses(tracks, measurements, validation_matrix):
    """
    Generates all possible association hypotheses.

    Args:
        tracks: A list of TrackObject instances.
        measurements: A NumPy array of measurements.
        validation_matrix: A NumPy array indicating valid track-measurement associations.

    Returns:
        A list of lists, where each inner list represents a hypothesis. 
        Example: [[0, 1], [2, 0]] -> Track 1 has no association, 
                                     Track 2 is associated with measurement 2. 
    """
    tracks_possible_associations = calculate_track_associations(tracks, measurements, validation_matrix)
    all_hypotheses = list(itertools.product(*tracks_possible_associations))
    return all_hypotheses

def calculate_jpda_probabilities(tracks, measurements, validation_matrix, pd, pg, beta_fa):
    all_hypotheses = generate_hypotheses(tracks, measurements, validation_matrix)  

    def calculate_hypothesis_probability(tracks, measurements, hypothesis, pd, pg, beta_fa):
        probability_factors = []
        for i, track_index in enumerate(hypothesis):
            track = tracks[i]
            if track_index > 0:  # Track is associated with a measurement
                meas_index = track_index - 1
                measurement = measurements[meas_index]

                innovation = measurement - track.get_output()
                S = track.get_innovation_covariance()
                rv = multivariate_normal(mean=np.zeros(S.shape[0]), cov=S)  # Assuming Gaussian
                probability_factors.append(pd * rv.pdf(innovation) / (1 - pd * pg))
            else:  # Track is not associated with a measurement
                probability_factors.append(1 - pd * pg) 

        num_false_alarms = 0
        for track_index in hypothesis:
            if track_index == 0:  # Check for no association
                num_false_alarms += 1

        return np.prod(probability_factors) * (beta_fa ** num_false_alarms) 

    hypothesis_probabilities = [calculate_hypothesis_probability(tracks, measurements, h, pd, pg, beta_fa) for h in all_hypotheses]

    num_tracks = len(tracks)
    num_meas = len(measurements)
    jpda_probs = np.zeros((num_tracks, num_meas + 1))
    tracks_possible_associations = calculate_track_associations(tracks, measurements, validation_matrix)  

    for i, track in enumerate(tracks):
        for j, meas_index in enumerate(tracks_possible_associations[i]):
            if meas_index > 0:  
                jpda_probs[i, meas_index - 1] = sum(hypothesis_probabilities[h_idx] for h_idx, h in enumerate(all_hypotheses) if h[i] == j)
            else:  
                jpda_probs[i, -1] = sum(hypothesis_probabilities[h_idx] for h_idx, h in enumerate(all_hypotheses) if h[i] == j)


    return jpda_probs

def main(measurements):
    global time
    confirmed_tracks = track_holder.get_confirmed_tracks()
    candidate_tracks = track_holder.get_candidate_tracks()
    MakeTimeUpdate(confirmed_tracks)
    MakeTimeUpdate(candidate_tracks)
    validation_matrix = create_validation_matrix(confirmed_tracks + candidate_tracks, measurements, GATE_THRESHOLD)
    # hypotheses = generate_hypotheses(confirmed_tracks + candidate_tracks, measurements, validation_matrix)
    # print(hypotheses)
    jpda_probs = calculate_jpda_probabilities(confirmed_tracks + candidate_tracks, measurements, validation_matrix, P_DET, P_GATE, BETA_FA)
    print(jpda_probs)
    if len(measurements) > 0:
        InitializeTracks(candidate_tracks, measurements) 
    time += 1
    if time % 3 == 0:
        track_holder.kill_all_tracks()
    # track_holder.kill_confirmed_tracks()
    # track_holder.kill_candidate_tracks()
    # track_holder.confirm_candidate_tracks()
    print("Candidate Tracks : ",len(candidate_tracks))
    print("Confirmed Tracks : ",len(confirmed_tracks))

track_holder = TrackHolder()
tracks_hist = []
for i in range(2):
    main(all_measurements[i])
tracks = track_holder.get_old_tracks()

for track in tracks:
    history = track.get_history()
    tracks_hist.append(history)
for track in track_holder.get_confirmed_tracks():
    history = track.get_history()
    tracks_hist.append(history)
tracks_hist = np.array(tracks_hist, dtype=object)
np.save('tracks',tracks_hist)

#     plt.plot(history[:,0], history[:,1], markersize = 5)
# plt.grid(True)
# plt.xlim((-3000,3000)),plt.ylim((-3000,3000))
# plt.title("GNN Tracker")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.legend(["Track 1", "Track 2", "Track 3", "Track 4", "Track 5"])
# plt.show()