import os
import cv2
import numpy as np
from utils import SparseDepth
from vision import Vision
from agentKinematics import RoboticAssistant


class IndoorDeepRL:
    #训练环境
    def __init__(self, map_path="complex.png"):
        #训练环境初始化
        self.terra = cv2.flip(cv2.imread(map_path), 0) #加载图片，读取到terra中
        #二值化图片
        self.terra[self.terra > 128] = 255
        self.terra[self.terra <= 128] = 0
        #图片归一化
        self.m = np.asarray(self.terra)
        self.m = cv2.cvtColor(self.m, cv2.COLOR_RGB2GRAY)
        self.m = self.m.astype(float) / 255.
        self.terra = self.terra.astype(float) / 255.
        self.lmodel = Vision(self.m) #实例化激光雷达

    def createInstance(self):
        #创建智能体(机器人)agent
        self.robot = RoboticAssistant(d=5, wu=9, wv=4, car_w=9, car_f=7, car_r=10, dt=0.1)
        #随机设置机器人的开始坐标和朝向
        self.robot.x, self.robot.y = self.random_start_travesable()
        self.robot.theta = 360 * np.random.random()
        self.pos = (self.robot.x, self.robot.y, self.robot.theta) #将机器人的坐标与朝向保存到pos参数中

        #随机设置终点坐标
        self.target = self.random_start_travesable()
        #计算距离和方向差
        self.target_euclidian = np.sqrt((self.robot.x - self.target[0]) ** 2 + (self.robot.y - self.target[1]) ** 2)
        target_angle = np.arctan2(self.target[1] - self.robot.y, self.target[0] - self.robot.x) - np.deg2rad(self.robot.theta)
        target_distance = [self.target_euclidian * np.cos(target_angle), self.target_euclidian * np.sin(target_angle)]

        self.sdata, self.plist = self.lmodel.measure_depth(self.pos)
        state = self.existance(self.sdata, target_distance)
        return state

    def step(self, action):
        #步进程序
        
        #速度进行限幅
        self.robot.control((action[0] + 1) / 2 * self.robot.v_interval, action[1] * self.robot.w_interval)
        #计算下一点位置
        self.robot.update()
        #计算四个边界点的坐标
        e1,e2,e3,e4 = self.robot.dimensions
        #查看四个边界点距离障碍物的距离，判断机器人是否撞上障碍物
        ee1 = SparseDepth(e1[0], e2[0], e1[1], e2[1])
        ee2 = SparseDepth(e1[0], e3[0], e1[1], e3[1])
        ee3 = SparseDepth(e3[0], e4[0], e3[1], e4[1])
        ee4 = SparseDepth(e4[0], e2[0], e4[1], e2[1])
        check = ee1+ee2+ee3+ee4 #将障碍物信息合成一个list
        
        #检查是否相撞
        collision = False
        for points in check:
            if self.m[int(points[1]),int(points[0])]<0.5:
                collision = True
                self.robot.redo()
                self.robot.velocity = -0.5 * self.robot.velocity
                break
        
        self.pos = (self.robot.x, self.robot.y, self.robot.theta) #更新当前机器人的坐标
        self.sdata, self.plist = self.lmodel.measure_depth(self.pos) #获取激光雷达的距离信息和终点坐标

        action_r = 0.05 if action[0] < -0.5 else 0 #奖励函数设置，然机器人运动的幅度大一点。
        
        curr_target_dist = np.sqrt((self.robot.x - self.target[0]) ** 2 + (self.robot.y - self.target[1]) ** 2)
        distance_reward = self.target_euclidian - curr_target_dist
   
        s_orien = np.rad2deg(np.arctan2(self.target[1] - self.robot.y, self.target[0] - self.robot.x))
        orientation_error = (s_orien - self.robot.theta) % 360
        if orientation_error > 180:
            orientation_error = 360 - orientation_error
        orientation_reward = np.deg2rad(orientation_error)
  
        reward = distance_reward - orientation_reward - 0.6 * action_r
        
        terminated = False
        
        if curr_target_dist < 20:
            reward = 20
            terminated = True
        if collision:
            reward = -15
            terminated = True
        
        self.target_euclidian = curr_target_dist
        target_angle = np.arctan2(self.target[1] - self.robot.y, self.target[0] - self.robot.x) - np.deg2rad(self.robot.theta)
        target_distance = [self.target_euclidian * np.cos(target_angle), self.target_euclidian * np.sin(target_angle)]
        state_next = self.existance(self.sdata, target_distance)

        return state_next, reward, terminated

    def render(self, gui=True):
        experiment_space = self.terra.copy()
        for pts in self.plist:
            cv2.line(
                experiment_space,
                (int(1*self.pos[0]), int(1*self.pos[1])), 
                (int(1*pts[0]), int(1*pts[1])),
                (0.0,1.0,0.0), 1)

        cv2.circle(experiment_space, (int(1*self.target[0]), int(1*self.target[1])), 10, (1.0,0.5,0.7), 3)
        experiment_space = self.robot.render(experiment_space)
        experiment_space = cv2.flip(experiment_space,0)
        if gui:
            cv2.imshow("Mapless Navigation",experiment_space)
            k = cv2.waitKey(1)

        return experiment_space.copy()
    
    def random_start_travesable(self):
        height, width = self.m.shape[0], self.m.shape[1]
        tx = np.random.randint(0,width)
        ty = np.random.randint(0,height)

        kernel = np.ones((10,10),np.uint8)  
        m_dilate = 1-cv2.dilate(1-self.m, kernel, iterations=3)
        while(m_dilate[ty, tx] < 0.5):
            tx = np.random.randint(0,width)
            ty = np.random.randint(0,height)
        return tx, ty
    
    def existance(self, sensor, target):
        si = [s/200 for s in sensor]
        ti = [t/500 for t in target]
        return si + ti

training = True
render = True
load_model = False
terrain = "map.png"
gif_path = "performance/"
model_path = "models/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
    
    
from PIL import Image

def performance_measure(episode, agent, s_count, max_rate):
     if episode>0 and episode%50==0:
        s_rate = s_count / 50
        if s_rate >= max_rate:
            max_rate = s_rate
            if training:
                print("Save model to " + model_path)
                agent.save_load_model("save", model_path)
        print("Success Rate (current/max):", s_rate, "/", max_rate)
     return max_rate

def visualize(agent, total_eps=2, message=False, render=False, map_path="large.png", gif_path="performance/", gif_name="test.gif"):
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    images = []

    mother_nature = IndoorDeepRL(map_path=terrain)
    for eps in range(total_eps):
        step = 0
        max_success_rate = 0
        success_count = 0

        state = mother_nature.createInstance()
        r_eps = []
        acc_reward = 0.

        while(True):
            action = agent.choose_action(state, eval=True)
            state_next, reward, terminated = mother_nature.step(action)
            displayed = mother_nature.render(gui=render)
            
            # 保存gif文件，此处注释掉加快运行速度
            # im_pil = Image.fromarray(cv2.cvtColor(np.uint8(displayed*255),cv2.COLOR_BGR2RGB))
            # images.append(im_pil)
            r_eps.append(reward)
            acc_reward += reward
            
            if message:
                print('\rEps: {:2d}| Step: {:4d} | action:{:+.2f}| R:{:+.2f}| Reps:{:.2f}  '\
                        .format(eps, step, action[0], reward, acc_reward), end='')
            
            state = state_next.copy()
            step += 1
            
            if terminated or step>200:
                if message:
                    print()
                break
    
    # 保存gif文件，此处注释掉加快运行速度
    # print("Create GIF ...")
    # if gif_path is not None:
        # images[0].save(gif_path+gif_name, save_all=True, append_images=images[1:], optimize=True, duration=40, loop=0)
        
import torch
import torch.nn as nn
import torch.nn.functional as F
import ddpg

batch_size = 64
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.layer1 = nn.Linear(23, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 2)

    def forward(self, s):
        hidden_layer_1 = F.relu(self.layer1(s))
        hidden_layer_2 = F.relu(self.layer2(hidden_layer_1))
        hidden_layer_3 = F.relu(self.layer3(hidden_layer_2))
        return torch.tanh(self.layer4(hidden_layer_3)) # one mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.layer1 = nn.Linear(23, 512)
        self.layer2 = nn.Linear(512+2, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, 1)
    
    def forward(self, s, a):
        hidden_layer_1 = F.relu(self.layer1(s))
        hidden_layer_1_a = torch.cat((hidden_layer_1, a), 1)
        hidden_layer_2 = F.relu(self.layer2(hidden_layer_1_a))
        hidden_layer_3 = F.relu(self.layer3(hidden_layer_2))
        return self.layer4(hidden_layer_3)


agent_mind_ddpg = ddpg.DDPG(base_net = [PolicyNet, QNet], b_size = batch_size)

if load_model:
    print("Load model ...", model_path)
    agent_mind_ddpg.save_load_model("load", model_path)

mother_nature = IndoorDeepRL(map_path=terrain)
total_steps = 0
max_success_rate = 0
success_count = 0
print("eps, step, total_steps, action[0], reward, loss_a, loss_c, agent_mind_ddpg.epsilon, acc_reward/step")

for eps in range(4500):
    state = mother_nature.createInstance()
    step = 0
    loss_a = loss_c = 0.
    acc_reward = 0.
    while True:
        if training:
            action = agent_mind_ddpg.choose_action(state, eval=False)
        else:
            action = agent_mind_ddpg.choose_action(state, eval=True)
        
        state_next, reward, terminated = mother_nature.step(action)
        end = 0 if terminated else 1
        agent_mind_ddpg.store_transition(state, action, reward, state_next, end)
        
        displayed = mother_nature.render(gui=render)

        loss_a = loss_c = 0.
        if total_steps > batch_size and training:
            loss_a, loss_c = agent_mind_ddpg.learn()
        step += 1
        total_steps += 1

        acc_reward += reward
        print('\r{:3d} ; {:4d}; {:6d}; {:+.2f}; {:+.2f}; {:+.2f}; {:+.2f}; {:.3f}; {:.2f}'
              .format(eps, step, total_steps, action[0], reward, loss_a, loss_c, agent_mind_ddpg._e, acc_reward/step), end='')
        state = state_next.copy()
        
        if terminated or step>200:
            if reward > 5:
                success_count += 1
            print()
            break

    max_success_rate = performance_measure(eps, agent_mind_ddpg, success_count, max_success_rate)
    print(max_success_rate)
    success_count = 0
    
    visualize(agent_mind_ddpg, total_eps=8, map_path=terrain, gif_path=gif_path, gif_name="DDPG_"+str(eps).zfill(4)+".gif")