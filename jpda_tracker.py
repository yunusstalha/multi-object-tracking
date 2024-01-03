
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


def calculate_path_probabilities(tracks, measurements, validation_matrix, total_tracks, current_column=1):
    """
    Calculate the probabilities of different paths based on track data and measurements.
    Parameters
    ----------
    tracks : list
        List of track objects.
    measurements : list 
        List of measurement data.
    validation_matrix : numpy.ndarray
        A matrix representing valid tracks.
    total_tracks : int 
        Total number of tracks.
    current_column : int
        Current column number in the validation matrix. Defaults to 1.

    Returns:
    ----------
    keys: list
        List of keys.
    path_probabilities: list
        List of path probabilities.
    Returns these as tuple (keys, path_probabilities)
    """
    path_probabilities = []
    keys_list = []
    temp_validation_matrix = validation_matrix.copy()

    if total_tracks == 0:
        return ([], [])
    if total_tracks == current_column:
        return _process_last_column(tracks, measurements, temp_validation_matrix, current_column)

    for measurement_index in range(len(temp_validation_matrix.T[current_column - 1])):
        updated_keys_list, updated_path_probs = _process_measurement(tracks, measurements, temp_validation_matrix, total_tracks, current_column, measurement_index)
        path_probabilities.extend(updated_path_probs)
        keys_list.extend(updated_keys_list)

    return keys_list, path_probabilities

def _process_last_column(tracks, measurements, validation_matrix, column_no):
    keys_list = []
    path_probs = []
    column = validation_matrix.T[column_no - 1]

    for measurement_index in range(len(column)):
       
        if measurement_index == 0:
       
            path_prob = BETA_FA * (1 - P_DET * P_GATE)
            keys_list.append([0])
            path_probs.append(path_prob)
       
        elif column[measurement_index] == 1:
       
            path_prob = _calculate_measurement_probability(tracks, measurements, column_no, measurement_index)
            keys_list.append([measurement_index])
            path_probs.append(path_prob)

    return keys_list, path_probs

def _process_measurement(tracks, measurements, validation_matrix, total_tracks, current_column, measurement_index):
    keys_list = []
    path_probs = []
    temp_validation_matrix = validation_matrix.copy()

    if measurement_index == 0:
       
        updated_keys_list, updated_path_probs = calculate_path_probabilities(tracks, measurements, temp_validation_matrix, total_tracks, current_column + 1)
       
        for path_index, path_prob in enumerate(updated_path_probs):
       
            keys_list.append([0] + updated_keys_list[path_index])
            path_probs.append(path_prob * (1 - P_DET * P_GATE))
   
    elif temp_validation_matrix.T[current_column - 1][measurement_index] == 1:
   
        temp_validation_matrix[measurement_index, :] = np.zeros((1, total_tracks))
        updated_keys_list, updated_path_probs = calculate_path_probabilities(tracks, measurements, temp_validation_matrix, total_tracks, current_column + 1)
   
        for path_index, path_prob in enumerate(updated_path_probs):
   
            measurement_prob = _calculate_measurement_probability(tracks, measurements, current_column, measurement_index)
            keys_list.append([measurement_index] + updated_keys_list[path_index])
            path_probs.append(path_prob * measurement_prob)

    return keys_list, path_probs

def _calculate_measurement_probability(tracks, measurements, column_no, measurement_index):
    
    rv = multivariate_normal([0,0,0,0], tracks[column_no - 1].get_innovation_covariance())
    innovation = measurements[measurement_index - 1] - tracks[column_no - 1].get_output()
    return BETA_FA * P_DET * rv.pdf(innovation)
    
def generate_validation_matrix(tracks, measurements):

    num_tracks = len(tracks)
    num_measurements = len(measurements)
    validation_matrix = np.zeros((num_measurements, num_tracks))
    validation_matrix = np.vstack((np.ones((1, num_tracks)), validation_matrix))
    for i in range(num_measurements):
        for j in range(num_tracks):
            if check_distance(tracks[j], measurements[i]) ** 2 < GATE_THRESHOLD:
                validation_matrix[i, j] = 1
    return validation_matrix

def associate(tracks, measurements, validation_matrix):
    """
    Associate tracks with measurements.
    
    Parameters
    ----------
    tracks : list
        List of track objects.
    measurements : list
        List of measurement data.
    validation_matrix : numpy.ndarray
        A matrix representing valid tracks.
        
    Returns
    -------
    association_list : list
        List of associations.    
    """
    number_of_measurements = len(measurements)
    number_of_tracks = len(tracks)

    path_keys, path_probabilities = calculate_path_probabilities(tracks, measurements, validation_matrix, number_of_tracks)
    association_list = []

    for track_index in range(number_of_tracks):
        track_associations = []
        associated_measurements = []
        associated_probs = []

        for key_index, key in enumerate(path_keys):
            measurement_index = key[track_index]
            probability = path_probabilities[key_index]

            if measurement_index in associated_measurements:
                existing_index = associated_measurements.index(measurement_index)
                associated_probs[existing_index] += probability
            else:
                associated_measurements.append(measurement_index)
                associated_probs.append(probability)
        
        track_associations.append([associated_measurements, associated_probs])
        association_list.append(track_associations)

    association_matrix = _create_association_matrix(tracks, measurements, number_of_measurements, number_of_tracks)

    return linear_sum_assignment(association_matrix)

def _create_association_matrix(tracks, measurements, num_measurements, num_tracks):
    association_matrix = np.ones((num_measurements, num_tracks + num_measurements)) * float('inf')

    for measurement_index in range(num_measurements):

        for track_index in range(num_tracks + num_measurements):
            if track_index < num_tracks:
                dist = check_distance(tracks[track_index], measurements[measurement_index]) ** 2
                if dist <= GATE_THRESHOLD:
                    association_matrix[measurement_index, track_index] = _calculate_log_probability(tracks[track_index], measurements[measurement_index])
            else:
                if measurement_index == track_index - num_tracks:
                    association_matrix[measurement_index, track_index] = -np.log(BETA_FA + BETA_NT)

    return association_matrix

def _calculate_log_probability(track, measurement):

    rv = multivariate_normal([0, 0, 0, 0], track.get_innovation_covariance())
    innovation = measurement - track.get_output()
    return -np.log(P_DET * rv.pdf(innovation) / (1 - P_DET * P_GATE))



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
    validation_matrix = generate_validation_matrix(all_tracks, measurements)
    associations = associate(all_tracks, measurements, validation_matrix)
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


# Initilazition
track_holder = TrackHolder()

for i in range(759):
# Tracker Loop
    main(all_measurements[i])


# Plotting
tracks = track_holder.get_old_tracks()
from matplotlib import pyplot as plt
for track in tracks:
    history = track.get_history()
    history = np.array(history)
    plt.plot(history[:,0], history[:,1], markersize = 5)
plt.grid(True)
plt.xlim((-3000,3000)),plt.ylim((-3000,3000))
plt.title("JPDAF Tracker")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend(["Track 1", "Track 2", "Track 3", "Track 4", "Track 5"])
plt.show()