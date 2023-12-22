
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
GATE_THRESHOLD = chi2.ppf(P_GATE, df=6) * 100
print(GATE_THRESHOLD)
all_measurements = np.load("/home/yunusi/git/multi-object-tracking/measurements.npy", allow_pickle=True)
def check_distance(kf_obj, meas):
    return mahalanobis(meas, kf_obj.get_output(), kf_obj.get_innovation_covariance())

def MakeTimeUpdate(tracks):
    for track in tracks:
        track.predict()
    return tracks

# def MakeMeasurementUpdate(tracks, measurements, associations):
#     track_assoc = []
#     row_list = associations[0]
#     column_list = associations[1]
#     for i in range (0, len(tracks)):
#         if i in column_list:
#             track_assoc.append(measurements[row_list[np.where(column_list == i)[0]],:])
#             np.delete(measurements, row_list[np.where(column_list == i)[0]], 0)

#         else:
#             track_assoc.append(None)
#     for i in range(len(column_list)):
#         np.delete(row_list, np.where(column_list == i)[0], 0)
#         np.delete(column_list, np.where(column_list == i)[0], 0)
#     for i in range(len(tracks)):
#         if track_assoc[i] is not None:
#             # print(track_assoc[i])
#             tracks[i].measurement_update(track_assoc[i])
#         else:
#             tracks[i].just_update()
            

def associate(tracks,measurements):

    num_tracks = len(tracks)
    num_meas = len(measurements)
    association_matrix = np.ones((num_meas, num_tracks + num_meas)) * float('inf') 

    for i in range(num_meas):
        for j in range(num_tracks + num_meas):
            if j < num_tracks:
                # print(, measurements[i])
                dist = check_distance(tracks[j], measurements[i])
                # print(dist)
                if dist < GATE_THRESHOLD:
                    rv = multivariate_normal([0,0,0,0], tracks[j].get_innovation_covariance())
                    innovation = measurements[i] - tracks[j].get_output()
                    association_matrix[i, j] = -np.log(P_DET * rv.pdf(innovation))
            else:
                if i == j - num_tracks:
                    association_matrix[i, j] = -np.log(BETA_FA + BETA_NT)
    # print(association_matrix)
    return linear_sum_assignment(association_matrix)

def InitializeTracks(tracks, measurements):
    for i in range(len(measurements)):
        mean = measurements[i]
        mean = np.array([mean[0], 0, mean[1], 0, mean[2], mean[3]])
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
    # InitializeTracks(candidate_tracks, measurements)
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
    plt.scatter(history[:,0], history[:,1])

plt.show()