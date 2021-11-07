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
        end = np.array(
            (robot_position[0] + self.farthest * np.cos(np.deg2rad(theta)), robot_position[1] + self.farthest * np.sin(np.deg2rad(theta))))
        x0, y0 = int(robot_position[0]), int(robot_position[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = SparseDepth(x0, x1, y0, y1)
        zone = self.farthest
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1] < 0 or p[0] < 0:
                continue
            if self.img_map[p[1], p[0]] < 0.5:
                aux = np.power(float(p[0]) - robot_position[0], 2) + np.power(float(p[1]) - robot_position[1], 2)
                aux = np.sqrt(aux)
                if aux < zone:
                    zone = aux
        return zone

    def measure_depth(self, current_pos):
        sense_data = []
        inter = (self.end_angle - self.start_angle) / (self.sensor_size - 1)
        for i in range(self.sensor_size):
            theta = current_pos[2] + self.start_angle + i * inter
            sense_data.append(self.line_of_sight(np.array((current_pos[0], current_pos[1])), theta))
        plist = distance_to_obstacle(current_pos, [self.sensor_size, self.start_angle, self.end_angle], sense_data)
        return sense_data, plist
