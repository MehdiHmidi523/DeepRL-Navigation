from utils import *
import numpy as np


class Vision:
    def __init__(self,
                 img_map,
                 sensor_size=21,
                 start_angle=-90.0,
                 end_angle=90.0,
                 max_dist=500.0,
                 ):
        self.sensor_size = sensor_size
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.farthest = max_dist
        self.img_map = img_map

    def line_of_sight(self, robot_position, theta):
        #计算激光雷达距离信息
        
        #计算每一条激光雷达线束的最远点的坐标，也就是线束终点坐标。
        end = np.array(
            (robot_position[0] + self.farthest * np.cos(np.deg2rad(theta)), robot_position[1] + self.farthest * np.sin(np.deg2rad(theta))))
        
        #设置激光雷达线的起点和终点坐标
        x0, y0 = int(robot_position[0]), int(robot_position[1])
        x1, y1 = int(end[0]), int(end[1])
        #将起点坐标和终点坐标传入到SparseDepth函数中，利用Bresenham算法生成一条直线，将坐标存储到plist中。
        plist = SparseDepth(x0, x1, y0, y1)
        zone = self.farthest #初始化距离变量zone
        
        #计算距离参数
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1] < 0 or p[0] < 0:
                #如果激光雷达的终点坐标已经超过地图的边界了，就不用计算长度了。
                continue
            if self.img_map[p[1], p[0]] < 0.5:
                #如果地图中当前点的坐标的颜色为黑色，就认为是碰到了障碍物。计算当前点到机器人坐标点的距离作为激光雷达的距离
                aux = np.power(float(p[0]) - robot_position[0], 2) + np.power(float(p[1]) - robot_position[1], 2)
                aux = np.sqrt(aux)
                if aux < zone:
                    zone = aux #将当前计算的距离放入到zone中
        return zone #返回激光雷达的距离信息

    def measure_depth(self, current_pos):
        #模拟激光雷达测距
        sense_data = [] #激光雷达深度数据
        inter = (self.end_angle - self.start_angle) / (self.sensor_size - 1) #确定每一步的步进角
        for i in range(self.sensor_size):
            theta = current_pos[2] + self.start_angle + i * inter #计算每条激光雷达线束的角度
            sense_data.append(self.line_of_sight(np.array((current_pos[0], current_pos[1])), theta)) #将每一条激光深度信息添加到sensor data中
        plist = distance_to_obstacle(current_pos, [self.sensor_size, self.start_angle, self.end_angle], sense_data) #获得激光雷达线束的终点坐标
        return sense_data, plist #返回激光雷达的传感器信息和终点坐标。
