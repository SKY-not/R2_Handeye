# 手眼标定系统

本项目用于 UR3 机械臂与 Intel RealSense D405 相机的手眼标定。当前支持 Eye-on-Hand 和 Eye-to-Hand 两种安装方式，支持棋盘格、单 AprilTag、四 AprilTag 标定板三种观测后端。

## 1. 硬件与依赖

硬件：

- 机械臂：UR3，通过 TCP/IP 通信，默认端口 `30003`
- 相机：Intel RealSense D405
- 标定目标：棋盘格、单 AprilTag、四 AprilTag 标定板

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 项目结构

```text
R2_Handeye/
├── main.py                  # 主入口
├── config.py                # 参数配置
├── device_manager.py        # 机器人和相机连接管理
├── data_collector.py        # 标定数据采集
├── calibration_solver.py    # 数据加载、AX=XB 初值、优化和保存
├── error_calculator.py      # 误差计算
├── result_visualizer.py     # 结果可视化
├── calibration/
│   ├── solver_axxb.py       # AX=XB SVD 求解
│   ├── optimizer.py         # 非线性优化
│   ├── transforms.py        # 坐标变换工具
│   └── feature_extractor.py # 棋盘格角点检测
├── camera/realsense.py      # RealSense D405 驱动
├── robot/ur_robot.py        # UR3 通信
├── data/                    # 标定数据
├── results/                 # 标定结果
└── tests/                   # 验证程序
```

## 3. 坐标与结果约定

标定结果保存在 `handeye_transform.txt`，记为 `X`。

| 模式 | 相机安装方式 | `X` 的含义 | 常用变换 |
| --- | --- | --- | --- |
| `eye_on_hand` | 相机安装在机械臂末端 | `T_tcp_camera` | `T_base_target = T_base_tcp @ X @ T_camera_target` |
| `eye_to_hand` | 相机固定在工作空间 | `T_base_camera` | `T_base_target = X @ T_camera_target` |

`T_camera_target` 是相机观测到的标定目标位姿：

- 棋盘格后端：目标是棋盘格坐标系
- 单 AprilTag 后端：目标是 tag 坐标系
- 四 AprilTag 标定板后端：目标是标定板中心坐标系

## 4. 相机内参

正常运行 `main.py` 时，程序使用当前 RealSense 相机实时读取到的内参和畸变参数：

```text
camera.intrinsics
camera.dist_coeffs
```

`config.py` 中的 `REALSENSE_CONFIG['default_intrinsics']` 只是兜底值。只有求解器没有收到真实相机内参时，才会使用默认内参。

如果选择“使用已有数据”，程序仍会连接当前相机，并使用当前相机读取到的内参重新计算已有图像数据。因此已有数据应尽量来自同一台相机、同一分辨率和同一套去畸变配置。

## 5. 标定流程

1. 运行主程序：

```bash
python main.py
```

2. 选择标定模式：`eye_on_hand` 或 `eye_to_hand`。
3. 选择观测后端：棋盘格、单 AprilTag、四 AprilTag 标定板。
4. 选择新采集数据，或使用 `data/{mode}` 下已有数据。
5. 每帧保存机械臂 TCP 位姿和相机观测到的目标位姿。
6. 使用所有帧两两组合构造 AX=XB 相对运动方程，通过 SVD 求解初值。
7. 以 SVD 结果为初值，对 6 自由度手眼矩阵做非线性优化。
8. 保存结果，计算误差，并可视化结果。

优化阶段只优化手眼矩阵 `X`。`z_scale` 固定为 `1.0`，`depth_scale.txt` 仅为兼容旧结果读取逻辑保留。

## 6. 观测后端

### 6.1 棋盘格

棋盘格角点由 `calibration/feature_extractor.py` 检测，位姿由 OpenCV `solvePnP` 根据棋盘格几何尺寸和相机内参估计。

相关配置：

```python
CHECKERBOARD_CONFIG = {
    'size': (11, 8),
    'square_size': 0.006,
    'board_to_base_rough': [...],
    'board_to_tcp_rough': [...],
}
```

`board_to_base_rough` 和 `board_to_tcp_rough` 只用于误差评估和可视化，不参与手眼矩阵求解。

### 6.2 单 AprilTag

单 AprilTag 后端使用 `pyapriltags` 自带的 `pose_R / pose_t` 估计 `T_camera_tag`，不使用 OpenCV `solvePnP`。

相关配置：

```python
APRILTAG_CONFIG = {
    'family': 'tag36h11',
    'tag_size': 0.03,
    'target_tag_id': 1,
    'decision_margin_threshold': 20.0,
    'min_area_ratio': 0.0005,
}
```

### 6.3 四 AprilTag 标定板

四 AprilTag 标定板后端同样使用 `pyapriltags` 自带 pose。每帧先对可见 tag 得到 `T_camera_tag_i`，再根据板上布局换算为板中心位姿：

```text
T_camera_board_i = T_camera_tag_i @ inverse(T_board_tag_i)
```

同一帧内如果识别到多个 tag，则先融合成一个 `T_camera_board`，再进入原有 AX=XB 标定流程。

融合方式：

- 平移：按 `decision_margin` 加权平均
- 旋转：使用 `scipy.spatial.transform.Rotation.mean()` 加权平均
- 如果只识别到一个有效 tag，则直接使用该 tag 推出的板中心位姿

板中心坐标系：

```text
原点：70 mm x 70 mm 标定板中心
x：向右
y：向下
z：垂直平面朝内
```

相关配置：

```python
APRILTAG_BOARD_CONFIG = {
    'family': 'tag36h11',
    'tag_size': 0.020,
    'tag_ids': [1, 2, 3, 4],
    'tag_centers': {
        1: [-0.015, -0.015, 0.0],
        2: [0.015, -0.015, 0.0],
        3: [-0.015, 0.015, 0.0],
        4: [0.015, 0.015, 0.0],
    },
    'board_to_tcp_rough': [0.00162, 0.0, 0.115, 0.0, -90.0, 0.0],
}
```

`board_to_tcp_rough` 是 Eye-to-Hand 情况下板中心相对于 TCP 的粗略位姿，单位为米和角度，仅用于误差评估和可视化。

## 7. 输出结果

结果保存在 `results/{mode}/`：

```text
results/{mode}/
├── handeye_transform.txt  # 4x4 手眼矩阵 X
├── depth_scale.txt        # 固定为 1.0，仅兼容保留
└── calibration_info.txt   # 模式、z_scale 和矩阵文本
```

实际使用时优先读取 `handeye_transform.txt`。

## 8. 误差与可视化

当前误差包括：

- 位置参考误差：将每帧观测到的目标位姿转换到参考坐标系后，与粗略配置位姿比较
- 旋转参考误差：与粗略配置姿态比较，单位为 degree
- 空间一致性误差：不依赖手动参考位姿，统计同一目标在基座系或 TCP 系下是否稳定
- 棋盘格重投影误差：棋盘格后端使用，单位为 pixel
- AprilTag 观测位姿重投影误差：使用 AprilTag 自身 `pose_R / pose_t` 将 tag 角点投回图像
- AprilTag 全链路重投影误差：通过 `手眼矩阵 + 机器人位姿 + 空间一致性平均目标位姿` 预测每帧 tag 角点，再投回图像

位置参考误差和旋转参考误差依赖粗略位姿准确性，主要用于参考。若参考位姿不准，误差可能偏大，但不一定代表手眼矩阵错误。

可视化内容包括：

- 3D 坐标系显示：base、TCP、camera、target reference、target estimated
- 位置和旋转误差分布图
- 参考标定板中心坐标系下的 `dx/dy/dz` 与 `dRx/dRy/dRz` 分量统计，用于判断哪个方向偏差最大
- 空间一致性平均目标位姿下的误差分量统计
- 棋盘格逐帧重投影角点对比

## 9. 验证

`tests/` 下的脚本用于实际验证标定结果，不属于主标定流程。

当前主要验证脚本：

```text
tests/move_tcp_eye_to_hand_apriltag.py
```

该脚本会读取 `handeye_transform.txt`，结合 AprilTag 观测和目标位姿配置，执行实际运动验证。标定是否可用，最终应以实际机器人闭环验证为准。

## 10. 数据采集建议

- 至少采集 6 帧，建议采集更多帧。
- 机械臂姿态变化要充分，尤其需要明显的旋转变化。
- 避免所有采样点集中在很小空间范围内。
- 标定目标应覆盖不同图像区域和不同距离。
- 四 AprilTag 标定板每帧至少识别到 1 个有效 tag；识别到多个 tag 时，帧内位姿更稳定。
- 如果某几帧检测质量明显差，建议删除后重新求解。
- 使用已有数据重新求解时，确认相机、分辨率、内参和去畸变配置与采集时一致。

## 11. 可选优化方向

- RGB-D 重投影优化：结合 RealSense 深度图恢复棋盘格角点 3D 点，再构建重投影残差进行非线性优化。
- RGB-D 棋盘格位姿估计：直接用角点像素和深度图恢复棋盘格 3D 角点，拟合棋盘格平面与坐标系，作为 `solvePnP` 之外的观测方式。
- 异常帧诊断：在现有误差分量统计基础上增加离群帧自动提示，辅助定位质量较差的采集帧。
