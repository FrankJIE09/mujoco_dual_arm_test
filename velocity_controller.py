# -*- coding: utf-8 -*-
"""
@File    :   velocity_controller.py
@Time    :   2024/07/28
@Author  :   Your Name
@Version :   1.0
@Desc    :   一个简单的基于P控制器的关节速度控制器。
"""
import numpy as np

class VelocityController:
    """
    一个简单的关节速度P控制器。
    根据期望关节位置和当前关节位置的误差，计算输出速度。
    """
    def __init__(self, kp):
        """
        初始化速度控制器。

        参数:
            kp (float or array-like): 比例增益。如果是单个值，则应用于所有关节。
                                     如果是一个数组，则每个元素对应一个关节的增益。
        """
        self.kp = np.array(kp)
        print(f"速度控制器已初始化, P增益 (Kp) = {self.kp}")

    def compute_velocity(self, q_desired, q_current):
        """
        计算伺服控制的速度指令。

        速度 = Kp * (期望位置 - 当前位置)

        参数:
            q_desired (np.ndarray): 期望的关节位置。
            q_current (np.ndarray): 当前的关节位置。

        返回:
            np.ndarray: 计算出的关节速度指令。
        """
        position_error = q_desired - q_current
        velocity_command = self.kp * position_error
        return velocity_command 