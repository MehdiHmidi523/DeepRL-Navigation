import numpy as np


def SparseDepth(x0, x1, y0, y1):
    #获得激光雷达线的所有点
    #使用Bresenham算法，具体的解释可以自行学习
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0 #给一个提前量，避免反应时间过短
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0 #给一个提前量，避免反应时间过短
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec


def distance_to_obstacle(robot_position, robot_sight_param, sensor_data):
    #计算激光雷达的终点坐标
    depth_awareness = []
    inter = (robot_sight_param[2] - robot_sight_param[1]) / (robot_sight_param[0] - 1)
    for i in range(robot_sight_param[0]):
        theta = robot_position[2] + robot_sight_param[1] + i * inter
        depth_awareness.append(
            [robot_position[0] + sensor_data[i] * np.cos(np.deg2rad(theta)), robot_position[1] + sensor_data[i] * np.sin(np.deg2rad(theta))])
    return depth_awareness

