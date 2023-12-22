
import numpy as np


class KalmanFilter(object):
    def __init__(self, 
                initial_state, 
                inital_state_covariance, 
                state_transition_mat, 
                observation_mat,
                process_noise_cov, 
                sensor_noise_cov):
        """
        Parameters
        ----------
        initial_state : numpy.array (nx1)
            Initial state of the system.
        inital_covariance : numpy.array (nxn)
            Initial covariance matrix of the system.
        state_transition_mat : numpy.array (nxn)
            State transition matrix.
        observation_mat : numpy.array (mxn)
            Observation matrix.
        process_noise_cov : numpy.array (nxn)
            Process noise covariance matrix.
        sensor_noise_cov : numpy.array (mxm)
            Sensor noise covariance matrix.
        """
        global TrackID
        self.state_transition_mat = state_transition_mat
        self.observation_mat = observation_mat
        self.process_noise_cov = process_noise_cov
        self.sensor_noise_cov= sensor_noise_cov
        self.state = initial_state 
        self.state_covariance = inital_state_covariance 
        self.innovation_covariance = np.zeros((self.observation_mat.shape[0], self.observation_mat.shape[0])) # mxm
        self.predicted_output = None
        self.time_alive = 0


    def predict(self):
        """
        Process update step of the Kalman filter.
        Paramteres
        ----------
        None
        Returns
        -------
        None
        """
        self.state = self.state_transition_mat @ self.state
        self.state_covariance = self.state_transition_mat @ self.state_covariance@ self.state_transition_mat.transpose() + self.process_noise_cov
        self.innovation_covariance = self.observation_mat @ self.state_covariance @ self.observation_mat.transpose() + self.sensor_noise_cov
        self.predicted_output = self.observation_mat @ self.state
        self.time_alive = self.time_alive + 1

    def measurement_update(self, measurement):
        """
        Measurement update step of the Kalman filter.
        Paramteres
        ----------
        measurement : numpy.array (mx1)
            Measurement vector.
        Returns
        ---------
        None
        """
        # print(measurement.shape)
        measurementt = np.array([measurement[:,0], measurement[:,1], measurement[:,2], measurement[:,3]])
        measurementt = measurementt.squeeze()
        kalman_gain = self.state_covariance @ self.observation_mat.transpose() @ np.linalg.inv(self.innovation_covariance)
        # print(self.state.shape, kalman_gain.shape, measurementt.shape, self.predicted_output.shape)
        self.state = self.state + kalman_gain @ (measurementt - self.predicted_output)
        self.state_covariance = self.state_covariance - kalman_gain @ self.innovation_covariance @ kalman_gain.transpose()

        
    # def pda_update(self,association):
    #     meas=association[0]
    #     meas_probs=association[1]
    #     K = self.P @ self.C.transpose() @ np.linalg.inv(self.inn_cov)
    #     updated_x_list=[]
    #     for m in range(len(meas)):
    #         updated_cov=self.P - K @ self.inn_cov @ K.transpose()
    #         if meas[m] == 0:
    #             y_eq=meas_probs[m]*self.y_pred
    #             sigma_u=meas_probs[m]*self.P
    #             updated_x_list.append(self.X)
    #         else:
    #             y_eq=y_eq+(meas_probs[m]*meas[m])
    #             sigma_u=sigma_u+meas_probs[m]*(updated_cov)
    #             updated_x_list.append(self.X + K @ (meas[m] - self.y_pred))
    #     self.X = self.X + K @ (y_eq - self.y_pred)
    #     sigma_spread=0
    #     for m in range(len(meas)):
    #         innov=updated_x_list[m]-self.X
    #         sigma_spread = sigma_spread + meas_probs[m] * innov @ innov.transpose()
    #     self.P=sigma_u+sigma_spread