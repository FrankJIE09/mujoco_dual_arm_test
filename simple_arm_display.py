"""
简单的双臂信息展示系统

使用pygame实时显示双臂末端执行器信息
- 位置、姿态、关节角度
- 中文界面显示
- 自适应屏幕分辨率

作者: frank
日期: 2024年12月
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
import time
import threading
import pygame
import sys
import os
import math
from dual_arm_kinematics import get_kinematics
from font_config import list_available_chinese_fonts, get_preferred_fonts


class SimpleArmDisplay:
    """
    简单的双臂信息展示器
    """

    def __init__(self, xml_file='dual_jaka_zu12_scene.xml'):
        """
        初始化展示器
        """
        self.xml_file = xml_file
        self.running = False

        print("正在初始化双臂信息展示器...")

        # 获取运动学模型
        self.left_chain, self.right_chain, self.T_W_B1, self.T_W_B2 = get_kinematics(xml_file)

        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # 初始化pygame
        pygame.init()

        # 获取显示器分辨率并设置窗口大小
        self.setup_display()

        # 设置中文字体
        self.setup_chinese_font()

        # 定义颜色
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 100, 255)
        self.GREEN = (0, 150, 0)
        self.RED = (200, 0, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_BLUE = (173, 216, 230)
        self.LIGHT_GREEN = (144, 238, 144)
        self.LIGHT_RED = (255, 200, 200)

        # 数据存储
        self.left_pose_data = {}
        self.right_pose_data = {}

        print("展示器初始化完成")
        print(f"窗口分辨率: {self.screen_width}x{self.screen_height}")

    def setup_display(self):
        """
        设置显示器和窗口大小
        """
        # 获取显示器信息
        info = pygame.display.Info()
        screen_width = info.current_w
        screen_height = info.current_h

        print(f"检测到显示器分辨率: {screen_width}x{screen_height}")

        # 根据显示器分辨率设置窗口大小（占屏幕的80%）
        self.screen_width = int(screen_width * 0.8)
        self.screen_height = int(screen_height * 0.8)

        # 确保最小尺寸
        self.screen_width = max(self.screen_width, 1000)
        self.screen_height = max(self.screen_height, 700)

        # 创建窗口
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("JAKA ZU12 双臂实时信息监控")

        # 计算字体大小（根据屏幕大小调整）
        base_size = min(self.screen_width, self.screen_height) // 50
        self.font_size_large = max(base_size + 4, 20)
        self.font_size_medium = max(base_size + 2, 18)
        self.font_size_small = max(base_size, 16)

    def test_font_chinese_support(self, font_path):
        """
        测试字体是否真正支持中文字符
        """
        try:
            test_font = pygame.font.Font(font_path, 24)
            test_text = "中文测试"
            test_surface = test_font.render(test_text, True, (0, 0, 0))

            # 检查渲染结果是否有实际内容
            if test_surface.get_width() > 50:  # 中文字符应该比较宽
                return True
            else:
                return False
        except Exception:
            return False

    def setup_chinese_font(self):
        """
        设置中文字体
        """
        print("正在配置中文字体...")

        # 首先尝试推荐的字体
        preferred_fonts = get_preferred_fonts()
        print(f"尝试推荐字体: {len(preferred_fonts)} 个")

        for font_path in preferred_fonts:
            if os.path.exists(font_path):
                print(f"测试推荐字体: {os.path.basename(font_path)}")
                if self.test_font_chinese_support(font_path):
                    try:
                        self.font_large = pygame.font.Font(font_path, self.font_size_large)
                        self.font_medium = pygame.font.Font(font_path, self.font_size_medium)
                        self.font_small = pygame.font.Font(font_path, self.font_size_small)
                        print(f"✅ 使用推荐中文字体: {os.path.basename(font_path)}")
                        return
                    except Exception as e:
                        print(f"  ❌ 字体加载失败: {e}")
                        continue
                else:
                    print(f"  ❌ 字体不支持中文: {os.path.basename(font_path)}")

        # 如果推荐字体都不行，搜索其他字体
        print("推荐字体不可用，搜索其他中文字体...")
        chinese_fonts = list_available_chinese_fonts()

        if chinese_fonts:
            print(f"找到 {len(chinese_fonts)} 个可能的中文字体，正在测试...")

            for i, font_path in enumerate(chinese_fonts):
                font_name = os.path.basename(font_path)
                print(f"  {i + 1}/{len(chinese_fonts)} 测试: {font_name}")

                if self.test_font_chinese_support(font_path):
                    try:
                        self.font_large = pygame.font.Font(font_path, self.font_size_large)
                        self.font_medium = pygame.font.Font(font_path, self.font_size_medium)
                        self.font_small = pygame.font.Font(font_path, self.font_size_small)
                        print(f"  ✅ 找到支持中文的字体: {font_name}")
                        return
                    except Exception as e:
                        print(f"  ❌ 字体加载失败: {e}")
                        continue
                else:
                    print(f"  ❌ 字体不支持中文: {font_name}")

        print("⚠️ 未找到支持中文的字体，安装中文字体包...")
        self.install_chinese_fonts()
        self.use_default_font()

    def install_chinese_fonts(self):
        """
        提示安装中文字体
        """
        print("\n" + "=" * 50)
        print("未找到支持中文的字体，建议安装中文字体包：")
        print("\nUbuntu/Debian系统:")
        print("  sudo apt update")
        print("  sudo apt install fonts-wqy-zenhei fonts-wqy-microhei")
        print("  sudo apt install fonts-noto-cjk fonts-noto-cjk-extra")
        print("\nCentOS/RHEL系统:")
        print("  sudo yum install wqy-zenhei-fonts wqy-microhei-fonts")
        print("  sudo yum install google-noto-cjk-fonts")
        print("\n安装完成后重新运行程序")
        print("=" * 50)

    def use_default_font(self):
        """
        使用默认字体
        """
        self.font_large = pygame.font.Font(None, self.font_size_large)
        self.font_medium = pygame.font.Font(None, self.font_size_medium)
        self.font_small = pygame.font.Font(None, self.font_size_small)

    def get_end_effector_pose(self, arm='left'):
        """
        获取末端执行器的当前姿态
        """
        # 获取当前关节角度
        if arm == 'left':
            current_q = self.data.qpos[6:12]  # 左臂使用后6个关节
            chain = self.left_chain  # 左臂使用left_chain
            base_transform = self.T_W_B1  # 左臂使用T_W_B1（left_base_transform）
        else:
            current_q = self.data.qpos[:6]  # 右臂使用前6个关节
            chain = self.right_chain  # 右臂使用right_chain
            base_transform = self.T_W_B2  # 右臂使用T_W_B2（right_base_transform）

        # 使用ikpy计算正向运动学
        fk_q = np.concatenate(([0], current_q, [0]))
        T_base_end = chain.forward_kinematics(fk_q)
        T_world_end = base_transform @ T_base_end

        # 提取位置
        position_world = T_world_end[:3, 3]
        position_base = T_base_end[:3, 3]

        # 提取姿态 - 世界坐标系
        rotation_matrix_world = T_world_end[:3, :3]
        rotation_world = Rotation.from_matrix(rotation_matrix_world)
        rpy_world_deg = rotation_world.as_euler('ZXY', degrees=True)

        # 提取姿态 - 基座坐标系
        rotation_matrix_base = T_base_end[:3, :3]
        rotation_base = Rotation.from_matrix(rotation_matrix_base)
        rpy_base_deg = rotation_base.as_euler('XYZ', degrees=True)

        return {
            'position_world': position_world,
            'position_base': position_base,
            'rpy_world_deg': rpy_world_deg,
            'rpy_base_deg': rpy_base_deg,
            'joint_angles_deg': current_q * 180.0 / math.pi,
            'timestamp': time.time()
        }

    def simulation_loop(self):
        """
        MuJoCo仿真循环（后台线程）
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 1)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while self.running and viewer.is_running():
                # 执行仿真步
                self.data.qvel = 10 * (self.data.ctrl - self.data.qpos)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # 更新姿态数据
                self.left_pose_data = self.get_end_effector_pose('left')
                self.right_pose_data = self.get_end_effector_pose('right')

                time.sleep(0.01)  # 10ms更新一次

        self.running = False

    def draw_text(self, text, x, y, font, color=None):
        """
        在指定位置绘制文本
        """
        if color is None:
            color = self.BLACK

        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
        return text_surface.get_height()

    def draw_rounded_rect(self, surface, color, rect, radius=10):
        """
        绘制圆角矩形
        """
        pygame.draw.rect(surface, color, rect, border_radius=radius)

    def draw_arm_info(self, arm_name, pose_data, start_x, start_y):
        """
        绘制单个机械臂的信息
        """
        if not pose_data:
            return

        y_offset = 0
        line_spacing = self.font_size_small + 5
        section_spacing = 15

        # 计算信息区域大小
        info_width = (self.screen_width // 2) - 40
        info_height = self.screen_height - start_y - 60

        # 绘制背景框
        bg_color = self.LIGHT_GREEN if arm_name == 'left' else self.LIGHT_RED
        self.draw_rounded_rect(self.screen, bg_color,
                               (start_x - 10, start_y - 10, info_width + 20, info_height), 15)

        # 标题
        title_color = self.GREEN if arm_name == 'left' else self.RED
        title_text = f"{arm_name.upper()}臂实时信息" if arm_name == 'left' else f"{arm_name.upper()}臂实时信息"
        y_offset += self.draw_text(title_text, start_x, start_y + y_offset,
                                   self.font_large, title_color) + section_spacing

        # 位置信息（世界坐标）
        pos = pose_data['position_world']
        y_offset += self.draw_text("位置 (世界坐标):", start_x, start_y + y_offset, self.font_medium, self.GREEN)
        y_offset += self.draw_text(f"  X: {pos[0]:.3f} 米", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  Y: {pos[1]:.3f} 米", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  Z: {pos[2]:.3f} 米", start_x, start_y + y_offset,
                                   self.font_small) + section_spacing

        # 位置信息（基座坐标）
        pos_base = pose_data['position_base']
        y_offset += self.draw_text("位置 (基座坐标):", start_x, start_y + y_offset, self.font_medium, self.GREEN)
        y_offset += self.draw_text(f"  X: {pos_base[0]:.3f} 米", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  Y: {pos_base[1]:.3f} 米", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  Z: {pos_base[2]:.3f} 米", start_x, start_y + y_offset,
                                   self.font_small) + section_spacing

        # 姿态信息（世界坐标系）
        rpy_world = pose_data['rpy_world_deg']
        y_offset += self.draw_text("RPY姿态 (世界坐标):", start_x, start_y + y_offset, self.font_medium, self.GREEN)
        y_offset += self.draw_text(f"  横滚角:  {rpy_world[0]:.1f}°", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  俯仰角:  {rpy_world[1]:.1f}°", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  偏航角:  {rpy_world[2]:.1f}°", start_x, start_y + y_offset,
                                   self.font_small) + section_spacing

        # 姿态信息（基座坐标系）
        rpy_base = pose_data['rpy_base_deg']
        y_offset += self.draw_text("RPY姿态 (基座坐标):", start_x, start_y + y_offset, self.font_medium, self.GREEN)
        y_offset += self.draw_text(f"  横滚角:  {rpy_base[0]:.1f}°", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  俯仰角:  {rpy_base[1]:.1f}°", start_x, start_y + y_offset, self.font_small) + 2
        y_offset += self.draw_text(f"  偏航角:  {rpy_base[2]:.1f}°", start_x, start_y + y_offset,
                                   self.font_small) + section_spacing

        # 关节角度
        joints = pose_data['joint_angles_deg']
        y_offset += self.draw_text("关节角度 (度):", start_x, start_y + y_offset, self.font_medium, self.GREEN)
        for i, angle in enumerate(joints):
            y_offset += self.draw_text(f"  关节{i + 1}: {angle:6.1f}°", start_x, start_y + y_offset,
                                       self.font_small) + 2

        return y_offset

    def draw_status_bar(self):
        """
        绘制状态栏
        """
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        status_text = f"状态: 运行中 | 时间: {current_time} | 按ESC键退出"

        # 绘制状态栏背景
        status_height = 35
        pygame.draw.rect(self.screen, self.GRAY,
                         (0, self.screen_height - status_height, self.screen_width, status_height))

        # 绘制状态文本
        self.draw_text(status_text, 10, self.screen_height - status_height + 8, self.font_small, self.WHITE)

    def draw_separator(self):
        """
        绘制分隔线
        """
        center_x = self.screen_width // 2
        pygame.draw.line(self.screen, self.GRAY,
                         (center_x, 60), (center_x, self.screen_height - 45), 3)

    def draw_header(self):
        """
        绘制标题栏
        """
        # 绘制标题背景
        header_height = 50
        pygame.draw.rect(self.screen, self.LIGHT_GREEN, (0, 0, self.screen_width, header_height))

        # 绘制标题
        title = "JAKA ZU12 双臂实时信息监控系统"
        title_surface = self.font_large.render(title, True, self.BLACK)
        title_x = (self.screen_width - title_surface.get_width()) // 2
        self.screen.blit(title_surface, (title_x, 15))

    def render(self):
        """
        渲染pygame界面
        """
        # 清空屏幕
        self.screen.fill(self.WHITE)

        # 绘制标题栏
        self.draw_header()

        # 绘制分隔线
        self.draw_separator()

        # 绘制左臂信息
        left_start_x = 20
        left_start_y = 70
        self.draw_arm_info("left", self.left_pose_data, left_start_x, left_start_y)

        # 绘制右臂信息
        right_start_x = self.screen_width // 2 + 20
        right_start_y = 70
        self.draw_arm_info("right", self.right_pose_data, right_start_x, right_start_y)

        # 绘制状态栏
        self.draw_status_bar()

        # 更新显示
        pygame.display.flip()

    def handle_events(self):
        """
        处理pygame事件
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True

    def run(self):
        """
        运行展示器主循环
        """
        self.running = True

        # 启动仿真线程
        sim_thread = threading.Thread(target=self.simulation_loop)
        sim_thread.daemon = True
        sim_thread.start()

        # pygame主循环
        clock = pygame.time.Clock()

        print("展示器已启动，按ESC键退出")

        while self.running:
            # 处理事件
            if not self.handle_events():
                break

            # 渲染界面
            self.render()

            # 控制帧率
            clock.tick(30)  # 30 FPS

        # 清理
        self.running = False
        pygame.quit()
        print("展示器已关闭")


def main():
    """
    主函数
    """
    print("启动双臂实时信息展示系统")
    print("=" * 40)

    try:
        # 创建展示器
        display = SimpleArmDisplay()

        # 运行展示器
        display.run()

    except Exception as e:
        print(f"程序运行出错: {e}")
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()
