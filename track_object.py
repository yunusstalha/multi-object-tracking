import numpy as np

from kalman import KalmanFilter

class TrackObject:
    def __init__(self,mean, track_id): 
        """
        Parameters
        ----------
        mean : numpy.array (nx1)
            Initial state of the system.
        covariance : numpy.array (nxn)
            Initial covariance matrix of the system.
        track_id : int
            Track ID of the object.
        """
        sensor_error = 50 #plus minus in meters, 3*sigma
        dt = 1
        initial_covariance = np.array([ [5**2, 0, 0, 0, 0, 0],
                                        [0, 1**2, 0, 0, 0, 0],
                                        [0, 0, 5**2, 0, 0, 0],
                                        [0, 0, 0, 1**2, 0, 0],
                                        [0, 0, 0, 0, 1**2, 0],
                                        [0, 0, 0, 0, 0, 1**2]])
        var_p=0.3
        x_var_process = var_p**2 
        y_var_process = var_p**2
        l_var_process = var_p**2 
        w_var_process = var_p**2 

        sigma_sensor = (sensor_error/3)
        sigma_sensor_box = 1

        state_transition_mat = np.array([[1, dt, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, dt, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

        gain_matrix = np.array([[dt**2/2,  0, 0,  0],
                                [dt,         0, 0, 0],
                                [0, dt**2/2, 0,    0],
                                [0, dt,      0,    0],
                                [0, 0,       1,    0],
                                [0, 0,       0,    1]])

        process_noise_cov = np.array([  [x_var_process, 0, 0, 0],
                                        [0, y_var_process, 0, 0],
                                        [0, 0, l_var_process, 0],
                                        [0, 0, 0, w_var_process]])

        process_noise_cov = gain_matrix @ process_noise_cov @ gain_matrix.transpose()

        observation_mat = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])

        sensor_noise_cov = np.array([[sigma_sensor**2, 0, 0, 0],
                                    [0,  sigma_sensor**2, 0, 0],
                                    [0,  0, sigma_sensor_box**2, 0],
                                    [0,  0, 0, sigma_sensor_box**2]])

        self.kf = KalmanFilter(mean, initial_covariance, state_transition_mat, observation_mat, process_noise_cov, sensor_noise_cov)
        self.track_id = track_id
        self.hit_vec = []
        self.history = []

    def predict(self):
        """
        Predict the state vector and the covariance matrix of the object.
        Parameters
        ----------
        None
        Returns
        -------
        None
        """
        self.kf.predict()

    def measurement_update(self, measurement):
        """
        Update the state vector and the covariance matrix of the object.
        Parameters
        ----------
        measurement : numpy.array (mx1)
            Measurement vector.
        Returns
        -------
        None
        """
        self.kf.measurement_update(measurement)
        self.hit_vec.append(1)
        self.history.append(measurement)

    def just_update(self):
        """
        Update the state vector and the covariance matrix of the object.
        Parameters
        ----------
        measurement : numpy.array (mx1)
            Measurement vector.
        Returns
        -------
        None
        """
        self.kf.predict
        self.hit_vec.append(0)
        self.history.append(self.kf.predicted_output)

    def get_state(self):
        """
        Returns the state vector of the object.
        Parameters
        ----------
        None
        Returns
        -------
        numpy.array (nx1)
            State vector of the object.
        """
        return self.kf.state
    
    def get_covariance(self):
        """
        Returns the covariance matrix of the object.
        Parameters
        ----------
        None
        Returns
        -------
        numpy.array (nxn)
            Covariance matrix of the object.
        """
        return self.kf.state_covariance
    
    def get_track_id(self):
        """
        Returns the track ID of the object.
        Parameters
        ----------
        None
        Returns
        -------
        int
            Track ID of the object.
        """
        return self.track_id
    def get_streak_vector(self):
        """
        Returns the streak vector of the object.
        Parameters
        ----------
        None
        Returns
        -------
        list
            Streak vector of the object.
        """
        return self.hit_vec