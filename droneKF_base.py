import numpy as np 
import math 
import matplotlib.pyplot as plt 

# using this for thrust variance 
default_var = 0.25 
# meas_var = np.random.uniform(0.01, 0.5) # m^2 
default_time_step = 1/200  # based on 200 Hz 

class DroneKF:
    def __init__(self, mass = 0.25, thrust = 2.7, thrust_var = default_var):
        self.mass = mass
        self.thrust = thrust 
        self.state = np.zeros((2,1))
        self.state_cov = np.identity(len(self.state)) * default_var
        self.pred_state = np.zeros_like(self.state)
        self.pred_state_cov = np.identity(len(self.state)) * default_var
        self.thrust_var = thrust_var
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
    
    # fold the G vector into the control itself
    # def G_vector(self, time_step = default_time_step, g=9.81):
    #     G = np.zeros((2,1))
    #     G[0] = -0.5 * g * (time_step**2)
    #     G[1] = -g * time_step
    #     return G 

    def C_matrix(self, time_step = default_time_step):
        C = np.array([[1, 0]]) # since we only measure the altitude and not velocity 
        return C 
    
    # KF prediction step
    def _predict(self, time_stamp, thrust, thrust_var=default_var):
        if self.time is None:
            raise RuntimeError("uninitialized filter")
        delta_t = time_stamp - self.time 
        self.time = time_stamp  

        A = self.A_matrix(delta_t)
        B = self.B_matrix(delta_t)
        C = self.C_matrix(delta_t)
        # G = self.G_vector(delta_t)

        # u = np.array([[thrust]])
        u = np.array([[thrust - self.mass * 9.81]])
        sigma_u = np.array([[thrust_var]])  
        Q = B @ sigma_u @ B.T

        # prediction 
        self.pred_state = A @ self.state + B @ u # + G
        self.pred_state_cov = A @ self.state_cov @ A.T + Q

        
    def _measure(self, z_alt, meas_var):
        C = self.C_matrix()
        # measurement noise covariance matrix 
        R = np.array([meas_var])
        
        # calculate Kalman Gain 
        SigmaXCT = self.pred_state_cov @ C.T 
        CSigmaXCT = C @ SigmaXCT 
        K = SigmaXCT @ np.linalg.inv(CSigmaXCT + R)

        # innovation 
        z = np.array([[z_alt]])
        self.state = self.pred_state + K @ (z - (C @ self.pred_state))
        self.state_cov = (np.eye(len(self.state)) - K @ C) @ self.pred_state_cov

    def _skip_measure(self):
        self.state = self.pred_state
        self.state_cov = self.pred_state_cov  # FIX: Keep predicted covariance instead of resetting

    def advance_filter(self, time_stamp, thrust, z_alt, meas_var, thrust_var=default_var):
        self._predict(time_stamp, thrust, thrust_var)  # FIX: Pass thrust_var through
        self._measure(z_alt, meas_var)

    def peek_altitude(self):
        return float(self.state[0,0])

    def peek_velocity(self):
        return float(self.state[1,0])

    def get_state(self):
        return self.state.copy(), self.state_cov.copy()

    def simulate_system(self, time_stamp):
        self._predict(time_stamp, self.thrust, self.thrust_var)
        self._skip_measure()

    def get_estimate(self):
        z = self.C_matrix() @ self.state
        z_cov = self.C_matrix() @ self.state_cov @ self.C_matrix().transpose()
        return z, z_cov


def main():
    sim_time = 5 # 5 seconds
    dt = default_time_step
    steps = int(sim_time / dt)
    t = np.arange(steps) * dt

    truth_env = DroneKF()
    truth_env.set_state(0.0, altitude=0.0, velocity=0.0)
    true_alt = np.zeros(steps)

    # part A 
    for k in range(steps):
        ts = k * dt  # time stamp
        truth_env.simulate_system(ts)
        true_alt[k] = truth_env.peek_altitude()

    # part B (no KF)
    R_stream = np.random.uniform(0.01, 0.5, size=steps)   # m^2
    alt_stream = true_alt + np.random.randn(steps) * np.sqrt(R_stream)
    
    face_alt = alt_stream.copy()

    # part C 
    kf_matched = DroneKF(mass=0.25, thrust=2.7, thrust_var=default_var)
    kf_matched.set_state(0.0, altitude=0.0, velocity=0.0)

    h_kf_matched = np.zeros(steps)
    for k in range(steps):
        ts = k * dt
        # use reported variance each step; pass through advance_filter
        kf_matched.advance_filter(ts, thrust=2.7, z_alt=alt_stream[k], 
                                   meas_var=R_stream[k], thrust_var=default_var)  
        h_kf_matched[k] = kf_matched.peek_altitude()

    # part D (mass is 10% higher)
    kf_mismatch = DroneKF(mass=0.25*1.10, thrust=2.7, thrust_var=default_var)
    kf_mismatch.set_state(0.0, altitude=0.0, velocity=0.0)

    h_kf_mismatch = np.zeros(steps)
    for k in range(steps):
        ts = k * dt
        kf_mismatch.advance_filter(ts, thrust=2.7, z_alt=alt_stream[k], 
                                    meas_var=R_stream[k], thrust_var=default_var)  
        h_kf_mismatch[k] = kf_mismatch.peek_altitude()

    # Plotting 
    fig1, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(t, true_alt,        label='(a) ground truth')
    ax1.plot(t, face_alt,         label='(b) sensor (face value)', alpha=0.6)
    ax1.plot(t, h_kf_matched,   label='(c) KF matched')
    ax1.plot(t, h_kf_mismatch,  label='(d) KF mass +10%')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('altitude [m]')
    ax1.set_title('Altitude: truth vs sensor vs KF')
    ax1.legend()
    ax1.grid(True)

    # 3 error traces on one figure
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(t, face_alt- true_alt,        label='sensor - truth', alpha=0.8)
    ax2.plot(t, h_kf_matched - true_alt,  label='KF matched - truth')
    ax2.plot(t, h_kf_mismatch - true_alt, label='KF +10% mass - truth')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('error [m]')
    ax2.set_title('Estimation errors')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()