#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('../scripts/')
from mcl import *
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse


# In[2]:


def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)                              # 共分散行列の固有値と固有ベクトル
    ang = math.atan2(eig_vec[:,0][1], eig_vec[:,0][0]) / math.pi*180
    return Ellipse(p, width=2*n*math.sqrt(eig_vals[0]), height=2*n*math.sqrt(eig_vals[1]), angle=ang,
                   fill=False, color="blue", alpha=0.5)

def matM(nu, omega, time, stds):
    return np.diag([stds["nn"]**2*abs(nu)/time + stds["no"]**2*abs(omega)/time, 
                    stds["on"]**2*abs(nu)/time + stds["oo"]**2*abs(omega)/time])

def matA(nu, omega, time, theta):
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega*time), math.cos(theta + omega*time)
    return np.array([[(stw-st)/omega,   -nu/(omega**2)*(stw-st) + nu/omega*time*ctw], 
                     [(-ctw+ct)/omega,  -nu/(omega**2)*(-ctw+ct) + nu/omega*time*stw],
                     [0,                 time]])

def matF(nu, omega, time, theta):
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu / omega * (math.cos(theta + omega * time) - math.cos(theta))
    F[1, 2] = nu / omega * (math.sin(theta + omega * time) - math.sin(theta))
    return F

def matH(pose, landmark_pos):
    mx, my = landmark_pos
    mux, muy, mut = pose
    q = (mux - mx)**2 + (muy - my)**2
    return np.array([[(mux - mx)/np.sqrt(q), (muy - my)/np.sqrt(q), 0.0],
                     [(my - muy)/q,          (mux - mx)/q,         -1.0]])

def matQ(distance_dev, direction_dev):
    return np.diag(np.array([distance_dev**2, direction_dev**2]))


# In[3]:


class KalmanFilter:
    def __init__(self, envmap, init_pose, motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}, 
                 distance_dev_rate=0.14, direction_dev=0.05):
        self.belief = multivariate_normal(mean=np.array([0.0, 0.0, 0.0]), cov=np.diag([1e-10, 1e-10, 1e-10])) # 信念分布
        self.motion_noise_stds = motion_noise_stds
        self.pose = self.belief.mean                                                # 信念分布の中心
        
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate                                  # 距離のセンサ値の標準偏差（距離をかける）
        self.direction_dev =  direction_dev                                         # 角度のセンサ値の標準偏差
        
    def motion_update(self, nu, omega, time):
        if abs(omega) < 1e-5:
            omega = 1e-5                                                            # ωがゼロのとき用        
        M = matM(nu, omega, time, self.motion_noise_stds)
        A = matA(nu, omega, time, self.belief.mean[2])
        F = matF(nu, omega, time, self.belief.mean[2])
        self.belief.cov = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T)       # 信念分布の共分散行列を更新（式(6.17)）
        self.belief.mean = IdealRobot.state_transition(nu, omega, time, self.belief.mean) # 信念分布の中心を更新
        self.pose = self.belief.mean
        
    def observation_update(self, observation):
        for d in observation:                                                       # 複数のランドマークの観測結果を一つずつ処理
            z = d[0]                                                                # センサ値
            obs_id = d[1]                                                           # ランドマークのID
            
            H = matH(self.belief.mean, self.map.landmarks[obs_id].pos)              # Hを求める
            estimated_z = IdealCamera.observation_function(self.belief.mean, self.map.landmarks[obs_id].pos) # h(μ)（観測前）を求める
            Q = matQ(estimated_z[0]*self.distance_dev_rate, self.direction_dev)     # Qを求める
            K = self.belief.cov.dot(H.T).dot(np.linalg.inv(Q + H.dot(self.belief.cov).dot(H.T))) # Kを求める
            self.belief.mean += K.dot(z - estimated_z)                              # 信念分布の中心を更新
            self.belief.cov = (np.eye(3) - K.dot(H)).dot(self.belief.cov)           # 信念分布の共分散行列を更新
            self.pose = self.belief.mean
    
    def draw(self, ax, elems):
        # xy平面上の誤差3σ範囲
        e = sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 3)      # XY平面での誤差楕円を描画(2次元だけ)
        elems.append(ax.add_patch(e))
        
        # θ方向の誤差の3σ範囲
        x, y, c = self.belief.mean                                                  # θの誤差3σ（平均値±3σ）を描画
        sigma3 = math.sqrt(self.belief.cov[2, 2])*3
        xs = [x + math.cos(c-sigma3), x, x + math.cos(c+sigma3)]
        ys = [y + math.sin(c-sigma3), y, y + math.sin(c+sigma3)]
        elems += ax.plot(xs, ys, color="blue", alpha=0.5)






