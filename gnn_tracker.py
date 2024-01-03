
import numpy as np
from track_holder import TrackHolder
from scipy.spatial.distance import mahalanobis
from scipy.stats.distributions import chi2
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from track_object import TrackObject

P_DET = 0.95
P_GATE = 0.997
BETA_FA = 6 / (3000 ** 2) 
BETA_NT = 3 / 760/(3000 ** 2)
GATE_THRESHOLD = chi2.ppf(P_GATE, df=6) * 1000


all_measurements = np.load("/home/yunusi/git/multi-object-tracking/measurements.npy", allow_pickle=True)

def check_distance(kf_obj, meas):
    return mahalanobis(meas, kf_obj.get_output(), kf_obj.get_innovation_covariance())

def MakeTimeUpdate(tracks):
    for track in tracks:
        track.predict()
    return tracks


def associate(tracks, measurements):
    """
    Associates tracks with measurements.

    :param tracks: A list of track objects.
    :param measurements: A list of measurement objects.
    :return: The result of linear sum assignment on the association matrix.
    """
    num_tracks = len(tracks)
    num_meas = len(measurements)
    
    # Initialize the association matrix with infinity
    association_matrix = np.full((num_meas, num_tracks + num_meas), np.inf)

    for meas_index in range(num_meas):
        for track_index in range(num_tracks + num_meas):
            if track_index < num_tracks:
                dist = check_distance(tracks[track_index], measurements[meas_index])
                
                if dist < GATE_THRESHOLD:
                    track = tracks[track_index]
                    innovation = measurements[meas_index] - track.get_output()
                    innovation_covariance = track.get_innovation_covariance()
                    rv = multivariate_normal([0, 0, 0, 0], innovation_covariance)
                    association_matrix[meas_index, track_index] = -np.log(P_DET * rv.pdf(innovation))
            else:
                if meas_index == track_index - num_tracks:
                    association_matrix[meas_index, track_index] = -np.log(BETA_FA + BETA_NT)

    return linear_sum_assignment(association_matrix)

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

def main(measurements):

    confirmed_tracks = track_holder.get_confirmed_tracks()
    candidate_tracks = track_holder.get_candidate_tracks()
    MakeTimeUpdate(confirmed_tracks)
    MakeTimeUpdate(candidate_tracks)

    all_tracks = track_holder.get_all_tracks()

    # print("All Tracks Len", len(all_tracks))
    associations = associate(all_tracks, measurements)
    # MakeMeasurementUpdate(confirmed_tracks, measurements, associations)
    # MakeMeasurementUpdate(candidate_tracks, measurements, associations)
    row_list = associations[0]
    column_list = associations[1]


    track_assoc = []
    for i in range (0, len(confirmed_tracks)):
        if i in column_list:
            track_assoc.append(measurements[row_list[np.where(column_list == i)[0]],:])
        else:
            track_assoc.append(None)
    for i in range(len(confirmed_tracks)):
        if track_assoc[i] is not None:
            # print(track_assoc[i])
            confirmed_tracks[i].measurement_update(track_assoc[i])
        else:
            confirmed_tracks[i].just_update()
    
    init_assoc = []
    for i in range (len(confirmed_tracks), len(confirmed_tracks) + len(candidate_tracks)):
        if i in column_list:
            init_assoc.append(measurements[row_list[np.where(column_list == i)[0]],:])
        else:
            init_assoc.append(None)

    for i in range(len(candidate_tracks)):
        if init_assoc[i] is not None:
            # print(track_assoc[i])
            candidate_tracks[i].measurement_update(init_assoc[i])
        else:
            candidate_tracks[i].just_update()
    
    for i in range (len(confirmed_tracks) + len(candidate_tracks), len(confirmed_tracks) + len(candidate_tracks) + measurements.shape[0]):
        if i in column_list:
            InitializeTracks(candidate_tracks,measurements[row_list[np.where(column_list == i)[0]],:])
    
    track_holder.kill_confirmed_tracks()
    track_holder.kill_candidate_tracks()
    track_holder.confirm_candidate_tracks()
    print("Candidate Tracks : ",len(candidate_tracks))
    print("Confirmed Tracks : ",len(confirmed_tracks))

track_holder = TrackHolder()
for i in range(759):
    main(all_measurements[i])
tracks = track_holder.get_old_tracks()
from matplotlib import pyplot as plt
for track in tracks:
    history = track.get_history()
    # print(history)
    history = np.array(history)
    # print(history.shape)
    plt.plot(history[:,0], history[:,1])
plt.grid(True)
plt.xlim((-3000,3000)),plt.ylim((-3000,3000))
plt.show()