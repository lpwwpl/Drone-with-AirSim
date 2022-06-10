# -*- coding:utf-8 -*-
"""
作者:高冲
日期:2021年09月09日
"""
from airsim import client
from airsim.types import ImageRequest, Quaternionr, Vector3r, VelocityControllerGains,PIDGains
import airsim
import time
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
from numba import jit
from scipy.optimize import curve_fit

nowx=[]
nowv=[]
# times=0
cx = 320  # u_0,像素系与图片系的x方向偏值，就是一半的图片像素
cy = 240  # v_0,像素系与图片系的y方向偏值，就是一半的图片像素
fx = 268.5  # 焦距/dx
fy = 268.5  # 焦距/dy     第六个圈时，相机与圆的差值变小了。


@jit
def get_circle_center_z2(mask, depthperspective):
    shape = depthperspective.shape
    circle_center_z = 0
    count = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if mask[i][j] != 0:
                k1 = (j - cx) / fx
                k2 = (i - cy) / fy
                z = depthperspective[i][j] / math.sqrt(k1 ** 2 + k2 ** 2 + 1)
                circle_center_z += z
                count += 1
    circle_center_z = circle_center_z / count
    return circle_center_z


class uav_setpoints:
    def __init__(self) -> None:
        self.circle_setpoint_moveToPositionAsync = (
            (-0.1, 3.5, -1.6),
            (0.7, 11.4, -0.90),
            (-0.8, 21.0, -1.1),
            (-11.5, 24.81, -0.85),#-11.6
            (-22.4, 25.3, -1.8),
            (-28.9, 23.0, -3.8)
        )

        self.circle_yaw_rotateToYawAsync = (
            90,
            90,
            135,
            180,
            180,
            180
        )

        self.land_setpoint_moveToPositionAsync = (-47.7, 20.4, -0.8)

    def get_circle_setpoint(self, id_from_one):
        return self.circle_setpoint_moveToPositionAsync[id_from_one - 1]

    def get_circle_yaw(self, id_from_one):
        return self.circle_yaw_rotateToYawAsync[id_from_one - 1]

    def get_land_setpoint(self):
        return self.land_setpoint_moveToPositionAsync


class circle_finder:
    def __init__(self, airsim_client) -> None:
        self.client = airsim_client
        self.cx = 320  # u_0,像素系与图片系的x方向偏值，就是一半的图片像素
        self.cy = 240  # v_0,像素系与图片系的y方向偏值，就是一半的图片像素
        self.fx = 268.5  # 焦距/dx
        self.fy = 268.5  # 焦距/dy

    def get_uav_position_rotation_in_wc(self):
        state = self.client.getMultirotorState()  # 获取无人机状态信息

        quaternionr = state.kinematics_estimated.orientation  # 姿态角
        w = quaternionr.w_val
        x = quaternionr.x_val
        y = quaternionr.y_val
        z = quaternionr.z_val
        tmp = [x, y, z, w]     # 四元数
        r = R.from_quat(tmp)   # 将四元数转换为旋转矩阵
        rotation_matrix = r.as_matrix()

        position = state.kinematics_estimated.position   # 无人机的位置
        position_list = []
        position_list.append(position.x_val)
        position_list.append(position.y_val)
        position_list.append(position.z_val)

        return position_list, rotation_matrix

    def get_depthperspective_image(self):
        # read_image = list(map(float, input("请输入1开始读照片").split()))[0]
        png_image = self.client.simGetImages([airsim.ImageRequest("2", airsim.ImageType.DepthPerspective,
                                                                  pixels_as_float=True, compress=False)])

        depthperspective = png_image[0]  # 8位浮点数格式单通道
        depthperspective = airsim.get_pfm_array(depthperspective)  # 将浮点数组变为pfm公式数组 (480,640)
        return depthperspective

    def get_circle_x_y(self,depthperspective):
        depthperspective[depthperspective > 5] = 0  # 让大于8的浮点数,置为0
        depthperspective = depthperspective.astype(np.uint8)  # 将float变为8位正整数
        depthperspective = cv2.equalizeHist(depthperspective)  # 对数据进行处理，有效的解决太亮或者太暗，提高图片对比
        # count= time.time()
        # cv2.imwrite(str(count)+".jpg",depthperspective)
        circle = [0, 0, 0]
        # 霍夫变换圆检测
        circles = None
        while circles is None:
            circles = cv2.HoughCircles(depthperspective, cv2.HOUGH_GRADIENT, 1,
                                       30, param1=None, param2=30, minRadius=30, maxRadius=300)  # 注意图片分辨率大小与圆半径检测
        circles = list(circles)  # 返回为N个圆的信息,[(1,N,3)]的格式
        circle += circles[0][0]

        return circle[0], circle[1]  # 输出检测到的圆的x, y坐标

    def get_circle_center_z(self, depthperspective):  # 获得z值
        mask = cv2.inRange(depthperspective, 1, 5)  # 低于8和高于8的变为0,其余的变为255
        circle_center_z = get_circle_center_z2(mask, depthperspective)
        return circle_center_z

    def circle_cc_to_wc(self, pixel_x, pixel_y, z, t, R):
        camera_inner_matrix = [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        camera_inner_matrix = np.linalg.pinv(np.array(camera_inner_matrix))  # 取逆  相机内参矩阵
        point2D_h = [pixel_x, pixel_y, 1]
        point = (np.array(point2D_h) * z).T
        tmp = np.dot(camera_inner_matrix, point)
        tmp[2] += 0.6  # 机身与相机的误差
        self.R_b_c = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]  # 相机系的Z变为机身系的X,X变为Y,Y变为Z,
        tmp = np.dot(self.R_b_c, tmp)

        result = np.dot(R, tmp) + np.array(t).T  # 乘旋转矩阵+移动向量
        print("circle in world frame,世界系,x,y,z", list(result))
        return list(result)

    def get_circle_position_in_wc(self):
        position_list, rotation_matrix = self.get_uav_position_rotation_in_wc()
        c=time.time()
        depthperspective = self.get_depthperspective_image()
        print("read_time: ",time.time()-c)
        circle_xy = self.get_circle_x_y(depthperspective)

        circle_z = self.get_circle_center_z(depthperspective)

        result = self.circle_cc_to_wc(circle_xy[0], circle_xy[1], circle_z, position_list, rotation_matrix)
        return result


class airsim_client:
    def __init__(self, ip_addr='127.0.0.1') -> None:
        print("Try to connect {}...".format(ip_addr))
        self.client = airsim.MultirotorClient(ip_addr)
        self.client.confirmConnection()
        self.client.enableApiControl(True)

        self.circle_finder = circle_finder(self.client)

        self.setpoints = uav_setpoints()

    def task_takeoff(self):
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        # self.client.moveByVelocityAsync(0, 0, -1, 1.5)
        self.client.hoverAsync().join()  # 悬停函数

    def task_to_1_2_3_circle(self, circle_id_from_one):
        dif_x, dif_y, dif_z = 100, 100, 100
        while (abs(dif_x) > 0.2 or abs(dif_y) > 0.2 or abs(dif_z) > 0.2) and circle_id_from_one <= 3:
            now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                            self.client.getMultirotorState().kinematics_estimated.position.y_val,
                            self.client.getMultirotorState().kinematics_estimated.position.z_val)
            next_position = self.setpoints.get_circle_setpoint(circle_id_from_one)[0:3]
            diff_position = (next_position[0] - now_position[0],
                             next_position[1] - now_position[1],
                             next_position[2] - now_position[2],
                             0.02)
            self.client.moveByVelocityAsync(*diff_position).join()
            arrived_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                self.client.getMultirotorState().kinematics_estimated.position.z_val)
            dif_x = next_position[0] - arrived_position[0]
            dif_y = next_position[1] - arrived_position[1]
            dif_z = next_position[2] - arrived_position[2]

    def task_to_1_2_3_circle1(self, circle_id_from_one):
        error_position = np.array([100, 100, 100])
        dt, p, d = 0.02, 1.0, 0.06
        next_position = np.array(self.setpoints.get_circle_setpoint(circle_id_from_one)[0:3])
        while abs(error_position[0]) > 0.5 or abs(error_position[1]) > 0.5 or abs(error_position[2]) > 0.5:

            now_position = np.array([self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                    self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                    self.client.getMultirotorState().kinematics_estimated.position.z_val])
            error_position = (next_position-now_position)
            now_velocity = np.array([self.client.getMultirotorState().kinematics_estimated.linear_velocity.x_val,
                                     self.client.getMultirotorState().kinematics_estimated.linear_velocity.y_val,
                                     self.client.getMultirotorState().kinematics_estimated.linear_velocity.z_val])
            velocitys = error_position * p + now_velocity * d
            self.client.setVelocityControllerGains(velocity_gains=VelocityControllerGains(x_gains=PIDGains(4, 0, 0.1),
                                                                                          y_gains=PIDGains(4, 0, 0.1),
                                                                                          z_gains=PIDGains(10, 4, 0.0)))
            self.client.moveByVelocityAsync(*velocitys, dt).join()

    def task_to_4_5_6_circle(self, circle_id_from_one):
        dif_x = 100
        dif_y = 100
        dif_z = 100
        next_position = self.setpoints.get_circle_setpoint(circle_id_from_one)[0:3]
        while (abs(dif_x) > 0.05 or abs(dif_y) > 0.08 or abs(dif_z) > 0.05) and circle_id_from_one <= 6:
            now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                            self.client.getMultirotorState().kinematics_estimated.position.y_val,
                            self.client.getMultirotorState().kinematics_estimated.position.z_val)
            diff_position = (next_position[0] - now_position[0],
                             next_position[1] - now_position[1],
                             next_position[2] - now_position[2],
                             0.01)
            self.client.moveByVelocityAsync(*diff_position).join()
            arrived_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                self.client.getMultirotorState().kinematics_estimated.position.z_val)
            dif_x = next_position[0] - arrived_position[0]
            dif_y = next_position[1] - arrived_position[1]
            dif_z = next_position[2] - arrived_position[2]

    def task_to_circle_1_3_moveByVelocityAsync(self, circle_id_from_one):
        # PID控制算法，位置环，来通过速度控制飞行
        error_position = np.array([100, 100, 100])
        dt, p, d = 0.02, 1.0, 0.06
        next_position = np.array(self.setpoints.get_circle_setpoint(circle_id_from_one)[0:3])
        while abs(error_position[0]) > 0.5 or abs(error_position[1]) > 0.5 or abs(error_position[2]) > 0.5:

            now_position = np.array([self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                    self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                    self.client.getMultirotorState().kinematics_estimated.position.z_val])
            error_position = (next_position-now_position)
            now_velocity = np.array([self.client.getMultirotorState().kinematics_estimated.linear_velocity.x_val,
                                     self.client.getMultirotorState().kinematics_estimated.linear_velocity.y_val,
                                     self.client.getMultirotorState().kinematics_estimated.linear_velocity.z_val])
            velocitys = error_position * p + now_velocity * d
            self.client.setVelocityControllerGains(velocity_gains=VelocityControllerGains(x_gains=PIDGains(4, 0, 0.1),
                                                                                          y_gains=PIDGains(4, 0, 0.1),
                                                                                          z_gains=PIDGains(10, 4, 0.0)))
            self.client.moveByVelocityAsync(*velocitys, dt).join()

    def task_to_circle_moveByVelocityAsync_PID(self, circle_id_from_one):
        # 位置环，来通过速度控制飞行
        error_position = np.array([100, 100, 100])
        dt = 0.01
        P = 1.0
        D = 0.04
        next_position = np.array(self.setpoints.get_circle_setpoint(circle_id_from_one)[0:3])
        c = time.time()
        while (abs(error_position[0]) > 0.2 or abs(error_position[1]) > 0.2 or abs(error_position[2]) > 0.2):
            now_position = np.array([self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                     self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                     self.client.getMultirotorState().kinematics_estimated.position.z_val])
            error_position = (next_position - now_position)
            velocitys = error_position * P  # +now_acceleration*D                   #+Intergration*I
            self.client.setVelocityControllerGains(
                velocity_gains=VelocityControllerGains(x_gains=PIDGains(4, 0, 0.08),
                                                       y_gains=PIDGains(4, 0, 0.08),
                                                       z_gains=PIDGains(10, 4, 1.2)))
            self.client.moveByVelocityAsync(*velocitys, dt).join()

    def task_cross_circle_1_3(self, circle_id_from_one):
        if circle_id_from_one < 4:
            circle_xyz = self.circle_finder.get_circle_position_in_wc()
            dif_x = 100
            dif_y = 100
            dif_z = 100
            while (abs(dif_x) > 0.3 or abs(dif_y) > 0.3 or abs(dif_z) > 0.3) and circle_id_from_one < 4:
                now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                self.client.getMultirotorState().kinematics_estimated.position.z_val)
                # circle_xyz = self.circle_finder.get_circle_position_in_wc()
                circle_diff_position = (circle_xyz[0] - now_position[0],
                                        circle_xyz[1] - now_position[1],
                                        circle_xyz[2] - now_position[2],
                                        0.1)
                self.client.moveByVelocityAsync(*circle_diff_position).join()
                Arrived_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                    self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                    self.client.getMultirotorState().kinematics_estimated.position.z_val)
                dif_x = circle_xyz[0] - Arrived_position[0]
                dif_y = circle_xyz[1] - Arrived_position[1]
                dif_z = circle_xyz[2] - Arrived_position[2]

    def task_cross_stick_moveByVelocityAsync(self):
        dif_x = 100
        dif_y = 100
        dif_z = 100
        next_position1 = [-40.8, 25.5, -3.2]
        while (abs(dif_x) > 0.2 or abs(dif_y) > 0.2 or abs(dif_z) > 0.2):
            now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                            self.client.getMultirotorState().kinematics_estimated.position.y_val,
                            self.client.getMultirotorState().kinematics_estimated.position.z_val)
            diff_position = (next_position1[0] - now_position[0],
                             next_position1[1] - now_position[1],
                             next_position1[2] - now_position[2],
                             0.01)
            self.client.moveByVelocityAsync(*diff_position).join()
            arrived_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                self.client.getMultirotorState().kinematics_estimated.position.z_val)
            dif_x = next_position1[0] - arrived_position[0]
            dif_y = next_position1[1] - arrived_position[1]
            dif_z = next_position1[2] - arrived_position[2]
        dif_x = 100
        dif_y = 100
        dif_z = 100
        next_position2 = [-40.8, 20.5, -1.5]
        while (abs(dif_x) > 0.2 or abs(dif_y) > 0.2 or abs(dif_z) > 0.2):
            now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                            self.client.getMultirotorState().kinematics_estimated.position.y_val,
                            self.client.getMultirotorState().kinematics_estimated.position.z_val)
            diff_position = (next_position2[0] - now_position[0],
                             next_position2[1] - now_position[1],
                             next_position2[2] - now_position[2],
                             0.01)
            self.client.moveByVelocityAsync(*diff_position).join()
            arrived_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                self.client.getMultirotorState().kinematics_estimated.position.z_val)
            dif_x = next_position2[0] - arrived_position[0]
            dif_y = next_position2[1] - arrived_position[1]
            dif_z = next_position2[2] - arrived_position[2]

    def move_cross_circle4(self, x, y, z):
        # 位置环，来通过速度控制飞行
        error_position = np.array([100, 100, 100])
        dt = 0.02
        P = np.array([2.0,2.0,10.0])
        # D = 0.06
        next_position = np.array([x, y, z])
        while ((error_position[0]) < 0.2 or abs(error_position[1]) > 0.3 or abs(error_position[2]) > 0.3):
            # c=time.time()
            now_position = np.array([self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                     self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                     self.client.getMultirotorState().kinematics_estimated.position.z_val])
            error_position = (next_position - now_position)

            velocitys = error_position * P# +now_velocity*D                   #+Intergration*I
            nowx.append([error_position[0], error_position[1], error_position[2]])
            # self.client.setVelocityControllerGains(
            #     velocity_gains=VelocityControllerGains(x_gains=PIDGains(4, 0, 0.08),
            #                                            y_gains=PIDGains(4, 0, 0.08),
            #                                            z_gains=PIDGains(10, 4, 1.2)))
            self.client.moveByVelocityAsync(*velocitys, dt).join()
            nowv.append([self.client.getMultirotorState().kinematics_estimated.linear_velocity.x_val,
                          self.client.getMultirotorState().kinematics_estimated.linear_velocity.y_val,
                          self.client.getMultirotorState().kinematics_estimated.linear_velocity.z_val])
            # print(time.time()-c)

    def move_cross_circle_5_6(self, x, y, z):
        # 位置环，来通过速度控制飞行
        error_position = np.array([100, 100, 100])
        dt = 0.02
        P = 1.0
        D = 0.04
        next_position = np.array([x, y, z])
        while ((error_position[0]) < 0.2 or abs(error_position[1]) > 0.3 or abs(error_position[2]) > 0.3):
            # c=time.time()
            now_position = np.array([self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                     self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                     self.client.getMultirotorState().kinematics_estimated.position.z_val])
            error_position = (next_position - now_position)

            velocitys = [error_position[0] * P*2, error_position[1] * P*2,
                         error_position[2] * P*2]  # +now_acceleration*D                   #+Intergration*I
            nowx.append([error_position[0], error_position[1], error_position[2]])

            self.client.moveByVelocityAsync(*velocitys, dt).join()
            nowv.append([self.client.getMultirotorState().kinematics_estimated.linear_velocity.x_val,
                         self.client.getMultirotorState().kinematics_estimated.linear_velocity.y_val,
                         self.client.getMultirotorState().kinematics_estimated.linear_velocity.z_val])
            # print(time.time()-c)

    def task_predict_circle_4(self, move_time=1.4, read_number=40):  # move_time用来弥补无人机移动需要的时间误差，read_number为采样数据个数
        def siny(x, a, b):  # 定义拟合的目标函数
            return 1.575 * np.sin(a * x + b) + 24.60
        # 读取相关数据
        first_t = time.time()
        t_list = []
        y_list = []
        read_time = []
        for i in range(read_number):
            before_read_time = time.time()
            circle_xyz = self.circle_finder.get_circle_position_in_wc()
            t_list.append(time.time()-first_t)
            read_time.append(time.time()-before_read_time)
            y_list.append(circle_xyz[1])  # 采样的数据为y轴坐标
        print("read_all_time", time.time()-first_t)
        t = np.mean(read_time)  # 平均读取图片时间
        find_y = np.array(y_list)
        find_t = np.array(t_list)/t  # 将读取图片的时间序列化，有利于拟合
        a_list = [-i * (2 * math.pi * t) / 40 for i in range(1, 11)] + \
                 [i * (2 * math.pi * t) / 40 for i in range(1, 11)]   # 10种不同速度下利于拟合出曲线的初始值

        a_pre_list = []     # 拟合出的曲线的a值
        b_pre_list = []     # 拟合出的曲线的b值
        pre_mean = []
        # 拟合曲线 2
        for a_first in a_list:  # 分别使用不同速度下设定的初始值进行拟合，平方误差小的为最佳拟合曲线
            popt, pcov = curve_fit(siny, find_t, find_y, [a_first, 0], maxfev=500000)
            a_pre_list.append(popt[0])
            b_pre_list.append(popt[1])
            yvals = siny(find_t, popt[0], popt[1])
            pre_mean.append(np.mean((find_y - yvals) ** 2))
        min_id = np.array(pre_mean).argmin()
        a = a_pre_list[min_id]   # 平方误差小的为最佳拟合曲线参数
        b = b_pre_list[min_id]

        # # plot curve
        # print("a", a)    # 5 0.063 1 0.019
        # print("b", b)
        # print(a_pre_list)
        # print(find_t.tolist())
        # # print(t_list)
        # print(find_y.tolist())
        # plt.plot(find_y)
        # plt.plot(siny(find_t, a, b))
        # plt.show()

        # predict Pos
        move_time=move_time/t
        print("move_time", move_time)
        read_time = find_t[-1]
        prey = siny((read_time+move_time), a, b)
        print("last_time", time.time()-first_t)
        return prey

    def task_predict_circle_5(self, move_time=1.5, read_number=60):
        def siny(y, a, b):
            return 1.14 * np.sin(a * y + b) + 25.20

        def sinz(z, a, b):
            return -(1.13 * np.sin(a * z + b) + 1.87)

        # 读取相关数据 1
        first_t=time.time()
        tlist=[]
        ylist=[]
        zlist=[]
        read_time=[]
        for i in range(read_number):
            before_read_time=time.time()
            circle_xyz = self.circle_finder.get_circle_position_in_wc()
            tlist.append(time.time()-first_t)
            read_time.append(time.time()-before_read_time)
            ylist.append(circle_xyz[1])
            zlist.append(circle_xyz[2])
        print("read_all_time",time.time()-first_t)
        t = np.mean(read_time)
        find_y = np.array(ylist)
        find_z = np.array(zlist)
        find_t = np.array(tlist)/t
        a_list = [-i * (2 * math.pi * t) / 40 for i in range(1, 11)] + [i * (2 * math.pi * t) / 40 for i in
                                                                        range(1, 11)]

        a1_pre_list, b1_pre_list, a2_pre_list, b2_pre_list, pre_mean1, pre_mean2 = [], [], [], [], [], []

        # 拟合曲线 2
        for a_first in a_list:
            # y_pre
            popt, pcov = curve_fit(siny, find_t, find_y, [a_first, 0], maxfev=500000)
            a1_pre_list.append(popt[0])
            b1_pre_list.append(popt[1])
            yvals = siny(find_t, popt[0], popt[1])
            # z_pre
            popt, pcov = curve_fit(sinz, find_t, find_z, [a_first, 0], maxfev=500000)
            a2_pre_list.append(popt[0])
            b2_pre_list.append(popt[1])
            zvals = sinz(find_t, popt[0], popt[1])
            # get_mean_std
            pre_mean1.append(np.mean((find_y - yvals) ** 2))
            pre_mean2.append(np.mean((find_z - zvals) ** 2))
        min_id1 = np.array(pre_mean1).argmin()
        min_id2 = np.array(pre_mean2).argmin()

        a1 = a1_pre_list[min_id1]
        b1 = b1_pre_list[min_id1]
        a2 = a2_pre_list[min_id2]
        b2 = b2_pre_list[min_id2]

        # 预测移动时间后，圆的位置 4
        move_time=move_time/t
        # print("move_time",move_time)
        read_time = find_t[-1]
        prey = siny((read_time+move_time), a1, b1)
        prez = sinz((read_time+move_time), a2, b2)
        print("last_time",time.time()-first_t)
        return prey, prez

    def task_predict_circle_6(self, move_time=1.5, read_number=60):
        def triangle_wave_y(x, a, b, c, T):
            y = np.where(np.mod(x - b, T) < T / 2, -4 / T * (np.mod(x - b, T)) + 1 + c / a, 0)
            y = np.where(np.mod(x - b, T) >= T / 2, 4 / T * (np.mod(x - b, T)) - 3 + c / a, y)
            return a * y

        def triangle_wave_z(x, a, b, c, T):
            z = np.where(np.mod(x - b, T) < T / 2, -4 / T * (np.mod(x - b, T)) + 1 + c / a, 0)
            z = np.where(np.mod(x - b, T) >= T / 2, 4 / T * (np.mod(x - b, T)) - 3 + c / a, z)
            return a * z

        first_t = time.time()
        read_time = []
        tlist = []
        ylist = []
        zlist = []
        current = 0  # 用来记录reda_time的下标
        count = 0  # 用来判断存储第一次读取的数值
        wait = 0  # 用来减小拟合函数时，没有读完整个半周期的误差

        before_circle_y = 0
        increase_count_y = 0
        decrease_count_y = 0
        y_max = 24.115
        y_min = 22.158
        pre_T_y = 0
        k_y = 1

        before_circle_z = 0
        increase_count_z = 0
        decrease_count_z = 0
        z_max = -2.764
        z_min = -4.75
        pre_T_z = 0
        k_z = 1

        for i in range(read_number):
            before_read_time = time.time()
            circle_xyz = self.circle_finder.get_circle_position_in_wc()
            tlist.append(time.time()-first_t)
            read_time.append(time.time()-before_read_time)

            if count == 0:
                before_circle_y = circle_xyz[1]
                before_circle_z = circle_xyz[2]
                count = count+1   # 第一次读取数据记录一次

            ylist.append(circle_xyz[1])
            if circle_xyz[1] > before_circle_y:
                increase_count_y += 1
                if decrease_count_y != 5:  # 如何解决抖的问题
                    decrease_count_y = 0
                if increase_count_y == 5:
                    k_y = (ylist[current]-ylist[current-5])/(tlist[current]-tlist[current-5])
                    pre_T_y = 2*(y_max-y_min)/k_y
            if circle_xyz[1] < before_circle_y:
                decrease_count_y += 1
                if increase_count_y != 5:  # 如何解决抖的问题
                    increase_count_y = 0
                if decrease_count_y == 5:
                    k_y = (ylist[current] - ylist[current - 5]) / (tlist[current] - tlist[current - 5])
                    pre_T_y = 2*(y_min - y_max) / k_y
            before_circle_y = circle_xyz[1]

            zlist.append(circle_xyz[2])
            if circle_xyz[2] > before_circle_z:
                increase_count_z += 1
                if decrease_count_z != 5:  # 如何解决抖的问题
                    decrease_count_z = 0
                if increase_count_z == 5:
                    k_z = (zlist[current]-zlist[current-5])/(tlist[current]-tlist[current-5])
                    pre_T_z = 2*(z_max-z_min)/k_z
            if circle_xyz[2] < before_circle_z:
                decrease_count_z += 1
                if increase_count_z != 5:  # 如何解决抖的问题
                    increase_count_z = 0
                if decrease_count_z == 5:
                    k_z = (zlist[current] - zlist[current - 5]) / (tlist[current] - tlist[current - 5])
                    pre_T_z = 2*(z_min - z_max) / k_z
            before_circle_z = circle_xyz[2]

            current += 1

        # print("read_all_time",time.time()-first_t)
        t = np.mean(read_time)
        pre_T_y = pre_T_y/t
        pre_T_z = pre_T_z / t

        find_y = np.array(ylist)
        find_z = np.array(zlist)
        find_t = np.array(tlist)/t
        # print("pre_T_y: ",pre_T_y)
        # print("pre_T_z: ", pre_T_z)

        popt, pcov = curve_fit(triangle_wave_y, find_t, find_y, [1, 0, 23.21, pre_T_y], maxfev=500000)
        a_y = popt[0]
        b_y = popt[1]
        c_y = popt[2]
        T_y = popt[3]
        # print("parm: ", a_y, b_y, c_y, T_y)
        popt, pcov = curve_fit(triangle_wave_z, find_t, find_z, [1, 0, -3.1, pre_T_z], maxfev=500000)
        a_z = popt[0]
        b_z = popt[1]
        c_z = popt[2]
        T_z = popt[3]
        # print("parm: ", a_z, b_z, c_z, T_z)

        move_time = move_time/t
        read_time = find_t[-1]
        prey = triangle_wave_y((read_time+move_time), a_y, b_y, c_y, T_y)
        prez = triangle_wave_z((read_time+move_time), a_z, b_z, c_z, T_z)
        print("last_time", time.time()-first_t)
        return prey, prez

    def task_to_by_v_land(self):
        dif_x, dif_y, dif_z = 100, 100, 100
        next_position = self.setpoints.get_land_setpoint()
        while abs(dif_x) > 0.3 or abs(dif_y) > 0.3 or abs(dif_z) > 0.3:
            now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                            self.client.getMultirotorState().kinematics_estimated.position.y_val,
                            self.client.getMultirotorState().kinematics_estimated.position.z_val)
            diff_position = (next_position[0] - now_position[0],
                             next_position[1] - now_position[1],
                             next_position[2] - now_position[2],
                             0.01)
            self.client.moveByVelocityAsync(*diff_position).join()
            arrived_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
                                self.client.getMultirotorState().kinematics_estimated.position.y_val,
                                self.client.getMultirotorState().kinematics_estimated.position.z_val)
            dif_x = next_position[0] - arrived_position[0]
            dif_y = next_position[1] - arrived_position[1]
            dif_z = next_position[2] - arrived_position[2]

    def task_land(self):
        self.client.hoverAsync().join()  # 悬停函数
        # time.sleep(1)
        self.client.landAsync().join()
        self.client.armDisarm(False)

    def begin_task(self):
        print("=========================")
        print("Taking off...")

        t = time.time()
        # Fly
        self.task_takeoff()

        # circle 1, 2, 3
        for circle_id_from_one in range(1, 3+1):
            self.task_to_1_2_3_circle(circle_id_from_one)
            self.client.hoverAsync().join()  # 悬停函数
            self.client.rotateToYawAsync(self.setpoints.get_circle_yaw(circle_id_from_one), margin=20).join()
            print(self.client.getMultirotorState().kinematics_estimated.orientation.z_val)
            self.task_cross_circle_1_3(circle_id_from_one)


        # circle 4
        print("=====================")
        print("going to circle {}".format(4))
        self.task_to_4_5_6_circle(4)
        # now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
        #                 self.client.getMultirotorState().kinematics_estimated.position.y_val,
        #                 self.client.getMultirotorState().kinematics_estimated.position.z_val)
        # print("now_position", now_position)
        self.client.hoverAsync().join()  # 悬停函数
        self.client.rotateToYawAsync(self.setpoints.get_circle_yaw(4), margin=1).join()
        # print(self.client.getMultirotorState().kinematics_estimated.orientation.z_val)
        print("=====================")
        print("predict the circle {}".format(4))
        c = time.time()
        y = self.task_predict_circle_4(move_time=1.35, read_number=40)  # 速度越大时，对移动时间精度的要求越高
        # y = self.task_predict_circle_4(move_time=1.7, read_number=40)  # 速度越大时，对移动时间精度的要求越高  read_time=0.5
        print("预测时间", time.time()-c)
        print("=====================")
        print("passing the circle {}".format(4))
        self.move_cross_circle4(-15.1, y, -0.90)
        print("预测+移动时间", time.time() - c)

        # circle 5
        print("=====================")
        print("going to circle {}".format(5))
        self.task_to_4_5_6_circle(5)
        # now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
        #                 self.client.getMultirotorState().kinematics_estimated.position.y_val,
        #                 self.client.getMultirotorState().kinematics_estimated.position.z_val)
        # print("now_position", now_position)
        self.client.hoverAsync().join()  # 悬停函数
        self.client.rotateToYawAsync(self.setpoints.get_circle_yaw(5), margin=1).join()
        # print(self.client.getMultirotorState().kinematics_estimated.orientation.z_val)
        print("=====================")
        print("predict the circle {}".format(5))
        c = time.time()
        y, z = self.task_predict_circle_5(move_time=1.45, read_number=40)  # 移动时间给得偏高一些比较合适
        # # y, z = self.task_predict_circle_5(move_time=1.7, read_number=40)  # 移动时间给得偏高一些比较合适 read_time=0.5
        print("预测时间", time.time()-c)
        print("=====================")
        print("passing the circle {}".format(5))
        self.move_cross_circle_5_6(-26.1, y, z)
        print("预测+移动时间", time.time() - c)

        # circle 6
        print("=====================")
        print("going to circle {}".format(6))
        self.task_to_4_5_6_circle(6)
        # now_position = (self.client.getMultirotorState().kinematics_estimated.position.x_val,
        #                 self.client.getMultirotorState().kinematics_estimated.position.y_val,
        #                 self.client.getMultirotorState().kinematics_estimated.position.z_val)
        # print("now_position", now_position)
        self.client.hoverAsync().join()  # 悬停函数
        # time.sleep(1.5)
        self.client.rotateToYawAsync(self.setpoints.get_circle_yaw(6), margin=1).join()
        # print(self.client.getMultirotorState().kinematics_estimated.orientation.z_val)
        print("=====================")
        print("predict the circle {}".format(6))
        # c = time.time()
        y, z = self.task_predict_circle_6(move_time=1.25, read_number=80)  # 移动时间给得偏高一些比较合适
        # print("预测时间", time.time() - c)
        print("=====================")
        print("passing the circle {}".format(6))
        self.move_cross_circle_5_6(-32.5, y, z)
        # print("预测+移动时间", time.time() - c)

        print("pass all circle time", time.time() - t)

        # 绕到角落飞过去
        self.task_cross_stick_moveByVelocityAsync()

        # 降落
        self.task_to_by_v_land()
        self.task_land()
        # print("alltime", time.time() - t)



if __name__ == "__main__":
    client = airsim_client('127.0.0.1')
    # 本地测试'192.168.1.1'
    client.begin_task()