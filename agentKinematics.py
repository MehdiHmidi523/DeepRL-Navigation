import cv2
import numpy as np


class RoboticAssistant:
    def __init__(self, v_range=60, w_range=90, d=5, wu=9, wv=4, car_w=9, car_f=7, car_r=10, dt=0.1):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.angular_velocity = 0
        self.record = []
        self.v_interval = v_range
        self.w_interval = w_range
        self.d = d
        self.wu = wu
        self.wv = wv
        self.car_w = car_w
        self.car_f = car_f
        self.car_r = car_r
        self.corps()
        self.delta_time = dt

    def update(self):
        if self.velocity > self.v_interval:
            self.velocity = self.v_interval
        elif self.velocity < -self.v_interval:
            self.velocity = -self.v_interval
        if self.angular_velocity > self.w_interval:
            self.angular_velocity = self.w_interval
        elif self.angular_velocity < -self.w_interval:
            self.angular_velocity = -self.w_interval
        self.x += self.velocity * np.cos(np.deg2rad(self.theta)) * self.delta_time
        self.y += self.velocity * np.sin(np.deg2rad(self.theta)) * self.delta_time
        self.theta += self.angular_velocity * self.delta_time
        self.theta = self.theta % 360
        self.record.append((self.x, self.y, self.theta))
        self.corps()

    def redo(self):
        self.x -= self.velocity * np.cos(np.deg2rad(self.theta)) * self.delta_time
        self.y -= self.velocity * np.sin(np.deg2rad(self.theta)) * self.delta_time
        self.theta -= self.angular_velocity * self.delta_time
        self.theta = self.theta % 360
        self.record.pop()

    def control(self, v, w):
        self.velocity = v
        self.angular_velocity = w
        if self.velocity > self.v_interval:
            self.velocity = self.v_interval
        elif self.velocity < -self.v_interval:
            self.velocity = -self.v_interval
        if self.angular_velocity > self.w_interval:
            self.angular_velocity = self.w_interval
        elif self.angular_velocity < -self.w_interval:
            self.angular_velocity = -self.w_interval

    def corps(self):
        p1 = change_direction_vis(self.car_f, self.car_w / 2, -self.theta) + np.array((self.x, self.y))
        p2 = change_direction_vis(self.car_f, -self.car_w / 2, -self.theta) + np.array((self.x, self.y))
        p3 = change_direction_vis(-self.car_r, self.car_w / 2, -self.theta) + np.array((self.x, self.y))
        p4 = change_direction_vis(-self.car_r, -self.car_w / 2, -self.theta) + np.array((self.x, self.y))
        self.dimensions = (p1.astype(int), p2.astype(int), p3.astype(int), p4.astype(int))

    def render(self, img=np.ones((600, 600, 3))):
        rm = 1000
        start = 0 if len(self.record) < rm else len(self.record) - rm

        color = (0 / 255, 97 / 255, 255 / 255)
        for i in range(start, len(self.record) - 1):
            cv2.line(img, (int(self.record[i][0]), int(self.record[i][1])),
                     (int(self.record[i + 1][0]), int(self.record[i + 1][1])), color, 1)

        ed1, ed2, ed3, ed4 = self.dimensions
        color = (0, 0, 0)
        size = 1
        cv2.line(img, tuple(ed1.astype(np.int).tolist()), tuple(ed2.astype(np.int).tolist()), color, size)
        cv2.line(img, tuple(ed1.astype(np.int).tolist()), tuple(ed3.astype(np.int).tolist()), color, size)
        cv2.line(img, tuple(ed3.astype(np.int).tolist()), tuple(ed4.astype(np.int).tolist()), color, size)
        cv2.line(img, tuple(ed2.astype(np.int).tolist()), tuple(ed4.astype(np.int).tolist()), color, size)

        arrow1 = change_direction_vis(6, 0, -self.theta) + np.array((self.x, self.y))
        arrow2 = change_direction_vis(0, 4, -self.theta) + np.array((self.x, self.y))
        arrow3 = change_direction_vis(0, -4, -self.theta) + np.array((self.x, self.y))
        cv2.line(img, (int(self.x), int(self.y)), (int(arrow1[0]), int(arrow1[1])), (0, 0, 1), 2)
        cv2.line(img, (int(arrow2[0]), int(arrow2[1])), (int(arrow3[0]), int(arrow3[1])), (1, 0, 0), 2)

        w1 = change_direction_vis(0, self.d, -self.theta) + np.array((self.x, self.y))
        w2 = change_direction_vis(0, -self.d, -self.theta) + np.array((self.x, self.y))
        img = view_unit(img, int(w1[0]), int(w1[1]), self.wu, self.wv, -self.theta)
        img = view_unit(img, int(w2[0]), int(w2[1]), self.wu, self.wv, -self.theta)
        img = cv2.line(img, tuple(w1.astype(np.int).tolist()), tuple(w2.astype(np.int).tolist()), (0, 0, 0), 1)
        return img


def change_direction_vis(x, y, angle):
    o = np.deg2rad(angle)
    return np.array((x * np.cos(o) + y * np.sin(o), -x * np.sin(o) + y * np.cos(o)))


def view_unit(rendered_unit, x, y, u, v, angle, color=(0, 0, 0), size=1):
    edge1 = change_direction_vis(-u / 2, -v / 2, angle) + np.array((x, y))
    edge2 = change_direction_vis(u / 2, -v / 2, angle) + np.array((x, y))
    edge3 = change_direction_vis(-u / 2, v / 2, angle) + np.array((x, y))
    edge4 = change_direction_vis(u / 2, v / 2, angle) + np.array((x, y))
    cv2.line(rendered_unit, tuple(edge1.astype(np.int).tolist()), tuple(edge2.astype(np.int).tolist()), color, size)
    cv2.line(rendered_unit, tuple(edge1.astype(np.int).tolist()), tuple(edge3.astype(np.int).tolist()), color, size)
    cv2.line(rendered_unit, tuple(edge3.astype(np.int).tolist()), tuple(edge4.astype(np.int).tolist()), color, size)
    cv2.line(rendered_unit, tuple(edge2.astype(np.int).tolist()), tuple(edge4.astype(np.int).tolist()), color, size)
    return rendered_unit
