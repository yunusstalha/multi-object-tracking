
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
print(GATE_THRESHOLD)
time = 0
all_measurements = np.load("/home/talha/git/multi-object-tracking/measurements.npy", allow_pickle=True)

def check_distance(kf_obj, meas):
    return mahalanobis(meas, kf_obj.get_output(), np.linalg.inv(kf_obj.get_innovation_covariance()))

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
                    association_matrix[meas_index, track_index] = -np.log(P_DET * rv.pdf(innovation) / (1 - P_DET * P_GATE))
            else:
                if meas_index == track_index - num_tracks:
                    association_matrix[meas_index, track_index] = -np.log(BETA_FA + BETA_NT)
    return linear_sum_assignment(association_matrix)

def update_tracks_and_remove_measurements(tracks, measurements, time):
    associations = associate(tracks, measurements)
    row_indices, col_indices = associations

    # Initialize a set to keep track of associated measurement indices
    associated_meas_indices = set()

    # Update tracks with associated measurements
    for col_index in col_indices:
        if col_index < len(tracks):
            meas_index = np.where(col_indices == col_index)[0][0]
            tracks[col_index].measurement_update(measurements[row_indices[meas_index]], time)
            associated_meas_indices.add(row_indices[meas_index])

    # Identify and update tracks without associated measurements
    for track_index, track in enumerate(tracks):
        if track_index not in col_indices:
            track.just_update(time)

    # Remove associated measurements from the measurements list
    measurements = np.delete(measurements, list(associated_meas_indices), axis=0)
    return measurements

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
    global time
    confirmed_tracks = track_holder.get_confirmed_tracks()
    candidate_tracks = track_holder.get_candidate_tracks()
    MakeTimeUpdate(confirmed_tracks)
    MakeTimeUpdate(candidate_tracks)

    # Update confirmed tracks and remove associated measurements
    measurements = update_tracks_and_remove_measurements(confirmed_tracks, measurements, time)

    # Update candidate tracks and remove associated measurements
    measurements = update_tracks_and_remove_measurements(candidate_tracks, measurements, time)

    # Initialization of New Tracks with remaining measurements
    if len(measurements) > 0:
        InitializeTracks(candidate_tracks, measurements)  # Assuming this function adds new tracks to the track holder

    # all_tracks = track_holder.get_all_tracks()

    # # print("All Tracks Len", len(all_tracks))
    # associations = associate(all_tracks, measurements)
    # # MakeMeasurementUpdate(confirmed_tracks, measurements, associations)
    # # MakeMeasurementUpdate(candidate_tracks, measurements, associations)
    # row_list = associations[0]
    # column_list = associations[1]


    # track_assoc = []
    # for i in range (0, len(confirmed_tracks)):
    #     if i in column_list:
    #         track_assoc.append(measurements[row_list[np.where(column_list == i)[0]],:])
    #     else:
    #         track_assoc.append(None)
    # for i in range(len(confirmed_tracks)):
    #     if track_assoc[i] is not None:
    #         # print(track_assoc[i])
    #         confirmed_tracks[i].measurement_update(track_assoc[i], time)
    #     else:
    #         confirmed_tracks[i].just_update(time)
    
    # init_assoc = []
    # for i in range (len(confirmed_tracks), len(confirmed_tracks) + len(candidate_tracks)):
    #     if i in column_list:
    #         init_assoc.append(measurements[row_list[np.where(column_list == i)[0]],:])
    #     else:
    #         init_assoc.append(None)

    # for i in range(len(candidate_tracks)):
    #     if init_assoc[i] is not None:
    #         # print(track_assoc[i])
    #         candidate_tracks[i].measurement_update(init_assoc[i], time)
    #     else:
    #         candidate_tracks[i].just_update(time)
    
    # for i in range (len(confirmed_tracks) + len(candidate_tracks), len(confirmed_tracks) + len(candidate_tracks) + measurements.shape[0]):
    #     if i in column_list:
    #         InitializeTracks(candidate_tracks,measurements[row_list[np.where(column_list == i)[0]],:])
    time += 1
    track_holder.kill_confirmed_tracks()
    track_holder.kill_candidate_tracks()
    track_holder.confirm_candidate_tracks()
    print("Candidate Tracks : ",len(candidate_tracks))
    print("Confirmed Tracks : ",len(confirmed_tracks))

track_holder = TrackHolder()
tracks_hist = []
for i in range(349):
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