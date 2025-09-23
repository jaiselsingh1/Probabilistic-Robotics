import numpy as np 
import math 

# using this for thrust variance 
default_var = 0.25 
# meas_var = np.random.uniform(0.01, 0.5) # m^2 
default_time_step = 1/200  # based on 200 Hz 
sim_time = 5 

class DroneKF:
    def __init__(self, mass: float = 0.25, thrust = 2.7, thrust_var = default_var):
        self.mass = mass
        self.thrust = thrust 
        self.state = np.zeros((2,1))
        self.state_cov = np.identity(len(self.state)) * default_var
        self.pred_state = np.zeros_like(self.state)
        self.pred_state_cov = np.identity(len(self.state)) * default_var
        self.measure_var = np.random.uniform(0.01, 0.5)
        self.time = None 

    def set_state(self, time_stamp, altitude, velocity):
        self.time = time_stamp 
        self.state[0, 0] = altitude 
        self.state[1, 0] = velocity 
        self.state_cov = np.identity(len(self.state)) * default_var

    def A_matrix(self, time_step = default_time_step):
        A = np.identity(len(self.state))
        A[0, 1] = time_step
        return A 

    def B_matrix(self, time_step = default_time_step):
        B = np.zeros((2,1))
        B[0] = (time_step)**2 / (2*self.mass)
        B[1] = time_step / self.mass
        return B 
    
    def G_vector(self, time_step = default_time_step, g=9.81):
        G = np.zeros((2,1))
        G[0] = -0.5 * g * (time_step**2)
        G[1] = -g * time_step
        return G 

    def C_matrix(self, time_step = default_time_step):
        C = np.array([1, 0]) # since we only measure the altitude and not velocity 
        return C 
    
    # KF prediction step
    def _predict(self, time_stamp, thrust, thust_var=default_var ):
        if self.time is None:
            raise RuntimeError("unitialized filter")
        delta_t = time_stamp - self.time 
        time_stamp = self.time 

        u = np.array([thrust])
        sigma_u = np.array([self.thrust_var])
        Q = B @ sigma_u @ B.T

        A = self.A_matrix(delta_t)
        B = self.B_matrix(delta_t)
        C = self.C_matrix(delta_t)
        G = self.G_vector(delta_t)

        # prediction 
        self.pred_state = A @ self.state + B * u + G
        self.pred_state_cov = A @ self.state_cov @ A.T + Q

        
    def _measure(self, z_alt):
        meas_var = self.meas_var 

        C = self.C_matrix()
        # measurement noise covariance matrix 
        R = np.array([meas_var])
        
        # calculate Kalman Gain 
        

    def _skip_measure(self):
        pass


    


def main():
    Drone = DroneKF()
