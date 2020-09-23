#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np


# In[2]:


class World:
    def __init__(self, time_span, time_interval, debug=False):                 # ここにロボットなどのオブジェクトを登録
        self.objects = []
        self.debug = debug
        self.time_span = time_span
        self.time_interval = time_interval
        
    def append(self, obj):              # オブジェクトを登録するための関数
        self.objects.append(obj)
        
    def draw(self):
        fig = plt.figure(figsize=(8, 8))     # 8*8 inchの図を準備
        ax = fig.add_subplot(111)            # サブプロットを準備
        ax.set_aspect('equal')               # 縦横比を座標の値と一致させる
        ax.set_xlim(-5, 5)                   # X軸を-5m * 5m の範囲で描画
        ax.set_ylim(-5, 5)                   # Y軸も同様に
        ax.set_xlabel("X", fontsize=20)      # X軸にラベルを表示
        ax.set_ylabel("Y", fontsize=20)      # 同じくY軸に
        
        elems = []
        
        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                         frames=int(self.time_span / self.time_interval) + 1, 
                                         interval=int(self.time_interval*1000), repeat=False)
            plt.show()
        
    def one_step(self, i, elems, ax):        # アニメーションを1コマ進める（ときに必要な処理）
        while elems:
            elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)


# In[3]:


class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        self.pose = pose               # 引数から姿勢の初期値を設定
        self.r = 0.2                   # これは描画のためなので固定値
        self.color = color             # 引数から描画するときの色を指定
        self.agent = agent             # エージェント
        self.poses = [pose]            # 軌跡の描画用
        self.sensor = sensor
        
    def draw(self, ax, elems):
        x, y, theta = self.pose             # 姿勢の変数を分解して3つの変数へ
        xn = x + self.r * math.cos(theta)   # ロボットの鼻先のx座標
        yn = y + self.r * math.sin(theta)   # ロボットの鼻先のy座標
        
        elems += ax.plot([x, xn], [y, yn], color=self.color)    # ロボットの向きを表す線分の描画
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)   # ロボットの円を表すオブジェクト
        elems.append(ax.add_patch(c)) # 円を描画
        
        self.poses.append(self.pose) # 軌跡の描画　poses[]に姿勢を追加
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)
    
    @classmethod
    def state_transition(cls, nu, omega, time, pose):   # nuはvt
        t0 = pose[2]                   # θt-1
        if math.fabs(omega) < 1e-10:   # fabsは絶対値、角速度がほぼゼロの場合とそうでない場合で場合分け
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0), 
                                     omega] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time] )
        
    def one_step(self, time_interval):
        if not self.agent: return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)


# In[4]:


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega
        
    def decision(self, observation=None):
        return self.nu, self.omega


# In[5]:


class LandMark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None
        
    def draw(self, ax, elems):
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))


# In[6]:


class Map:
    def __init__(self):
        self.landmarks = []
        
    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)     # landmarks[]の長さをidにして登録、追加するランドマークにIDを与える
        self.landmarks.append(landmark)
        
    def draw(self, ax, elems):                # IdealRobotのdrawと同じ引数、同じような処理（elemsに追加）を行う
        for lm in self.landmarks: lm.draw(ax, elems)


# In[7]:


class IdealCamera:
    def __init__(self, env_map, distance_range=(0.5, 6.0), direction_range=(-math.pi/3, math.pi/3)):  # 測定できる範囲
        self.map = env_map
        self.lastdata = []
        
        self.distance_range = distance_range
        self.direction_range = direction_range
        
    def visible(self, polarpos):
        if polarpos is None:
            return False
        
        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1]              and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]    # z(測定値)が範囲内にあればTrue(不等号)
        
    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))
            
        self.lastdata = observed
        return observed
    
    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= np.pi:
            phi -= 2*np.pi
        while phi < -np.pi:
            phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi] ).T
    
    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")


# In[8]:


if __name__ == "__main__":
    world = World(30, 0.1)

    # 地図を生成して3つランドマークを追加
    m = Map()
    m.append_landmark(LandMark(2, -2))
    m.append_landmark(LandMark(-1, -3))
    m.append_landmark(LandMark(3, 3))
    world.append(m)
                  
    # ロボットを作る
    straight = Agent(0.2, 0.0)                 # 0.2[m/s]で直進
    circling = Agent(0.2, 10.0/180*math.pi)    # 0.2[m/s], 10[deg]で円を描く
    robot1 = IdealRobot(np.array([2, 3, math.pi/6]).T,     sensor=IdealCamera(m), agent=straight)
    robot2 = IdealRobot(np.array([-2, -1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
    world.append(robot1)
    world.append(robot2)

    # アニメーション実行
    world.draw()



