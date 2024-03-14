import numpy as np
import matplotlib.pyplot as plt
num_clutters = 0
num_states = 6
num_obs_states = 4
num_max_objects = 3

# Process Class

class Process(object):
    def __init__(self, X, A, x_var, y_var, l_var, w_var, G):
        self.X = X
        self.A = A
        self.x_var = x_var
        self.y_var = y_var
        self.l_var = l_var
        self.w_var = w_var
        self.G = G
          
    def update(self):
        var = np.array([np.random.randn() * np.sqrt(self.x_var), np.random.randn() * np.sqrt(self.y_var), np.random.randn() * np.sqrt(self.l_var), np.random.randn() * np.sqrt(self.w_var)])
        self.X = self.A @ self.X + self.G @ var.transpose()
        return self.X
    
# Initial Params

sensor_error = 1 #plus minus in meters, 3*sigma
np.random.seed(7)
x0 = np.array([0, 1, 0, 1, 20, 25]) 
x0_2 = np.array([-250, 1, 0, 1, 30, 36]) 
x0_3 = np.array([-500, 1, -70, 1, 25, 28]) 
x0_4 = np.array([-500, 1, 0, 1,5, 2]) 
x0_5 = np.array([1000, 1, 0, 1, 5, 1]) 

P0 = np.array([[5**2, 0, 0, 0, 0, 0],
               [0, 1**2, 0, 0, 0, 0],
               [0, 0, 5**2, 0, 0, 0],
               [0, 0, 0, 1**2, 0, 0],
               [0, 0, 0, 0, 1**2, 0],
               [0, 0, 0, 0, 0, 1**2]])
var_p = 0.3
x_var_process = var_p**2 # (sigma_x)^2
y_var_process = var_p**2 # (sigma_y)^2
l_var_process = var_p**2 # (sigma_l)^2
w_var_process = var_p**2 # (sigma_w)^2

dt = 1 #timestep
steps = 350


sigma_sensor = (sensor_error/3)
sigma_sensor_box = 1

A = np.array([[1, dt, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

G = np.array([[dt**2/2, 0, 0, 0],
              [dt,0, 0, 0],
              [0, dt**2/2, 0, 0],
              [0, dt, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

Q_tilda = np.array([[x_var_process, 0, 0, 0],
                    [0, y_var_process, 0, 0],
                    [0, 0, l_var_process, 0],
                    [0, 0, 0, w_var_process]])

Q = G @ Q_tilda @ G.transpose()

C = np.array([[1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

R = np.array([[sigma_sensor**2, 0, 0, 0],
              [0,  sigma_sensor**2, 0, 0],
              [0,  0, sigma_sensor_box**2, 0],
              [0,  0, 0, sigma_sensor_box**2]])

duration = dt*steps

start1 = 0
duration1 = 250
start2 = 50
duration2 = 200
start3 = 100
duration3 = 250

# start4 = 50
# duration4 = 300

# start5 = 200
# duration5 = 108

# state_process1 = Process(x0.transpose(), A, x_var_process, y_var_process, l_var_process, w_var_process, G)
# state_process2 = Process(x0_2.transpose(), A, x_var_process, y_var_process, l_var_process, w_var_process, G)
# state_process3 = Process(x0_3.transpose(), A, x_var_process, y_var_process, l_var_process, w_var_process, G)
# # state_process4 = Process(x0_2.transpose(), A, x_var_process, y_var_process, l_var_process, w_var_process, G)
# # state_process5 = Process(x0_3.transpose(), A, x_var_process, y_var_process, l_var_process, w_var_process, G)
# X1 = np.full((num_states,duration), np.nan)
# X2 = np.full((num_states,duration), np.nan)
# X3 = np.full((num_states,duration), np.nan)
# # X4 = np.full((num_states,duration), np.nan)
# # X5 = np.full((num_states,duration), np.nan)

# Y1 = np.full((num_obs_states,duration-1), np.nan)
# Y2 = np.full((num_obs_states,duration-1), np.nan)
# Y3 = np.full((num_obs_states,duration-1), np.nan)
# # Y4 = np.full((num_obs_states,duration-1), np.nan)
# # Y5 = np.full((num_obs_states,duration-1), np.nan)

# for i in range (start1, start1+duration1):
#     X1[:,i] = state_process1.update()
    
# for i in range (start2, start2+duration2):
#     X2[:,i] = state_process2.update()
    
# for i in range (start3, start3+duration3):
#     X3[:,i] = state_process3.update()
 
# # for i in range (start4, start4+duration4):
# #     X4[:,i] = state_process4.update()
    
# # for i in range (start5, start5+duration5):
# #     X5[:,i] = state_process5.update()
    
# for i in range (start1, start1+duration1-1):
#     Y1[:,i] = [X1[0,i] + np.random.randn() * sigma_sensor, X1[2,i] + np.random.randn() * sigma_sensor, X1[4,i] + np.random.randn() * sigma_sensor_box, X1[5,i] + np.random.randn() * sigma_sensor_box]
    
# for i in range (start2+1, start2+duration2-1):
#     Y2[:,i] = [X2[0,i] + np.random.randn() * sigma_sensor, X2[2,i] + np.random.randn() * sigma_sensor, X2[4,i] + np.random.randn() * sigma_sensor_box, X2[5,i] + np.random.randn() * sigma_sensor_box]
    
# for i in range (start3+1, start3+duration3-1):
#     Y3[:,i] = [X3[0,i] + np.random.randn() * sigma_sensor, X3[2,i] + np.random.randn() * sigma_sensor, X3[4,i] + np.random.randn() * sigma_sensor_box, X3[5,i] + np.random.randn() * sigma_sensor_box]

# # for i in range (start4+1, start4+duration4-1):
# #     Y4[:,i] = [X4[0,i] + np.random.randn() * sigma_sensor, X4[2,i] + np.random.randn() * sigma_sensor, X4[4,i] + np.random.randn() * sigma_sensor_box, X4[5,i] + np.random.randn() * sigma_sensor_box]
    
# # for i in range (start5+1, start5+duration5-1):
# #     Y5[:,i] = [X5[0,i] + np.random.randn() * sigma_sensor, X5[2,i] + np.random.randn() * sigma_sensor, X5[4,i] + np.random.randn() * sigma_sensor_box, X5[5,i] + np.random.randn() * sigma_sensor_box]

# plt.plot(Y1[0],Y1[1])
# plt.plot(Y2[0],Y2[1])
# plt.plot(Y3[0],Y3[1])


# plt.show()
# np.save("y1", Y1)
# np.save("y2", Y2)
# np.save("y3", Y3)
# exit()

# ----------------------------------------------------------------
Y1 = np.load("y1.npy")
Y2 = np.load("y2.npy")
Y3 = np.load("y3.npy")
clutters_x = np.random.uniform(-1000, 1000, num_clutters)
clutters_y = np.random.uniform(1000,1000, num_clutters)
clutters_l = np.random.uniform(4, 20, num_clutters)
clutters_w = np.random.uniform(4, 30, num_clutters)

for i in range (2, steps):
    clutters_x = np.c_[clutters_x, np.random.uniform(-1000, 1000, num_clutters)]
    clutters_y = np.c_[clutters_y, np.random.uniform(-1000,1000, num_clutters)]
    clutters_l = np.c_[clutters_l, np.random.uniform(4, 20, num_clutters)]
    clutters_w = np.c_[clutters_w, np.random.uniform(4, 30, num_clutters)]
   

clutters_x = np.vstack([clutters_x, Y1[0,:]])
# clutters_x = np.vstack([clutters_x, Y2[0,:]])
# clutters_x = np.vstack([clutters_x, Y3[0,:]])

# clutters_x = np.vstack([clutters_x, Y4[0,:]])
# clutters_x = np.vstack([clutters_x, Y5[0,:]])

clutters_y = np.vstack([clutters_y, Y1[1,:]])
# clutters_y = np.vstack([clutters_y, Y2[1,:]])
# clutters_y = np.vstack([clutters_y, Y3[1,:]])

# clutters_y = np.vstack([clutters_y, Y4[1,:]])
# clutters_y = np.vstack([clutters_y, Y5[1,:]])

clutters_l = np.vstack([clutters_l, Y1[2,:]])
# clutters_l = np.vstack([clutters_l, Y2[2,:]])
# clutters_l = np.vstack([clutters_l, Y3[2,:]])

# clutters_l = np.vstack([clutters_l, Y4[2,:]])
# clutters_l = np.vstack([clutters_l, Y5[2,:]])

clutters_w = np.vstack([clutters_w, Y1[3,:]])
# clutters_w = np.vstack([clutters_w, Y2[3,:]])
# clutters_w = np.vstack([clutters_w, Y3[3,:]])

# clutters_w = np.vstack([clutters_w, Y4[3,:]])
# clutters_w = np.vstack([clutters_w, Y5[3,:]])
data_list = []
for k in range (0, steps - 1):
    x_meas = clutters_x[:,k]
    y_meas = clutters_y[:,k]
    l_meas = clutters_l[:,k]
    w_meas = clutters_w[:,k]
    x_meas = x_meas[~np.isnan(x_meas)]
    y_meas = y_meas[~np.isnan(y_meas)]
    l_meas = l_meas[~np.isnan(l_meas)]
    w_meas = w_meas[~np.isnan(w_meas)]
    data = np.array(np.array([x_meas, y_meas, l_meas, w_meas]).transpose())
    data_list.append(data)
data_list = np.array(data_list,dtype='object')
np.save('measurements',data_list)
import matplotlib.pyplot as plt
fig = plt.figure(constrained_layout=True, figsize=(15,15))
grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)

x_y = fig.add_subplot(grid[0,0])
plt.grid(True)
x_y.plot(clutters_x, clutters_y, marker='x', markersize = 2.5, linestyle = 'None', color = 'r')
plt.xlim((-3000,3000)),plt.ylim((-3000,3000))
plt.title("Clutterd Data")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()