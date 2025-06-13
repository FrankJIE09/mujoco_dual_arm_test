# URDF 到 MJCF 可靠转换工具

这是一个用于将URDF（Unified Robot Description Format）文件批量转换为MJCF（MuJoCo Physics Engine XML format）的工具。
该项目旨在解决在转换过程中因mesh文件路径问题导致的常见失败，提供一个稳定、可靠的转换流程。
目前主要支持Elfin、UR系列的机器人臂和RoHand机器人手，但可以轻松扩展以支持其他机器人。

## ✨ 主要特性

- **可靠的路径处理**：通过智能地处理mesh文件路径，确保MuJoCo编译器可以正确找到所有资源，从而极大地提高了转换成功率。
- **批量转换**：可以自动发现指定目录下的所有URDF文件并进行批量转换。
- **独立的输出**：为每个机器人模型创建一个独立的文件夹，其中包含转换后的MJCF文件和所有相关的mesh文件，方便分发和使用。
- **易于扩展**：代码结构清晰，可以方便地添加对新类型机器人的支持。
- **可视化支持**：内置一个简单的MJCF查看器 (`mjcf_viewer.py`)，方便快速验证转换结果。

## 📂 项目结构

```
.
├── mjcf_models/            # 存放转换后输出的MJCF模型
├── elfin_description/      # 存放Elfin系列机器人的URDF和mesh文件
├── ur_description/         # 存放UR系列机器人的URDF和mesh文件
├── rohand_urdf_ros2/       # 存放RoHand机器人手的URDF和mesh文件
├── urdf_converter.py       # 核心转换脚本
├── mjcf_viewer.py          # MJCF模型查看器
├── requirements.txt        # Python依赖
└── README.md
```

## 🚀 环境依赖

确保你的Python环境中安装了必要的库。

1.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    `requirements.txt`包含以下核心库：
    - `mujoco`: 核心物理引擎和转换工具。
    - `numpy`: 科学计算库。

## 💡 使用方法

### 1. 准备URDF文件

将你的机器人描述文件（URDF和meshes）放入相应的目录中。

- 对于Elfin机器人，请放入 `elfin_description` 目录。
- 对于UR机器人，请放入 `ur_description` 目录。
- 对于RoHand机器人手，请放入 `rohand_urdf_ros2` 目录。

> **注意**: 如果你的模型是`.xacro`格式，你需要先将其转换为`.urdf`。你可以参考 `xacro_to_urdf.sh` 脚本中的示例来完成这个步骤。通常使用`xacro`命令：
> ```bash
> xacro input.xacro > output.urdf
> ```

### 2. 执行转换

运行核心转换脚本来开始批量转换。

```bash
python urdf_converter.py
```

脚本会自动扫描 `elfin_description/urdf`、`ur_description/urdf` 和 `rohand_urdf_ros2/urdf` 目录下的所有`.urdf`文件，并执行转换。
转换成功后，结果将保存在 `mjcf_models/` 目录下。每个模型都会有一个独立的子目录，例如 `mjcf_models/elfin5/` 或 `mjcf_models/rohand_left/`。

### 3. 查看模型

使用项目提供的 `mjcf_viewer.py` 脚本来可视化和验证转换后的模型。

```bash
python mjcf_viewer.py mjcf_models/elfin5/elfin5.xml
```

例如，要查看 `elfin5` 模型：

```bash
python mjcf_viewer.py mjcf_models/elfin5/elfin5.xml
```

## 🧰 辅助脚本说明

本项目包含一些实用的辅助脚本，以简化开发流程。

- **`xacro_to_urdf.sh`**:
  - **功能**: 将`.xacro`文件批量或单独转换为`.urdf`文件。这是在运行主转换脚本 `urdf_converter.py` 之前一个重要的预处理步骤。
  - **依赖**: 需要ROS环境以及`xacro`包。
  - **用法**:
    ```bash
    # 转换单个模型 (例如: elfin5)
    ./xacro_to_urdf.sh elfin5

    # 交互式选择模型
    ./xacro_to_urdf.sh
    ```

- **`view_in_rviz.sh`**:
  - **功能**: 一键式脚本，用于在ROS的**RViz**中可视化机器人模型。它会自动处理XACRO到URDF的转换，并启动RViz。
  - **依赖**: 需要完整的ROS环境（包含RViz和roslaunch）。
  - **用法**:
    ```bash
    # 在RViz中查看elfin5模型
    ./view_in_rviz.sh elfin5
    ```

- **`generate_requirements.sh`**:
  - **功能**: 使用`pipreqs`自动扫描项目并生成`requirements.txt`文件，同时在文件头部注入当前的环境信息。
  - **依赖**: `pipreqs` (如果未安装，脚本会尝试自动安装)。
  - **用法**:
    ```bash
    ./generate_requirements.sh
    ```

## 📝 注意事项

- **Mesh文件格式**：脚本目前支持 `.stl`, `.dae`, `.obj`, `.ply` 格式的mesh文件。
- **ROS环境**: `view_in_rviz.sh` 和 `xacro_to_urdf.sh` 脚本需要在ROS环境中使用。
- **自定义机器人**: 如果要添加对新机器人的支持，你需要在 `urdf_converter.py` 脚本中：
    1.  在 `convert_robot` 函数中添加新的机器人类型逻辑，指定URDF和mesh源目录。
    2.  在 `main` 函数中更新 `robots_to_convert` 列表的生成逻辑。

---
*该README由AI编程助手生成。*
