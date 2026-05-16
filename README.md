# 手眼标定系统

基于 UR3 机械臂和 Intel RealSense D405 相机的手眼标定工具。当前主流程支持棋盘格和 AprilTag 两种观测后端，支持 Eye-on-Hand 和 Eye-to-Hand 两种安装方式。

## 项目结构

```text
R2_Handeye/
├── main.py                 # 主入口
├── config.py               # 参数配置
├── device_manager.py       # 机器人和相机连接管理
├── data_collector.py       # 标定数据采集
├── calibration_solver.py   # 数据加载、AX=XB 初值、优化和保存
├── error_calculator.py     # 位姿参考误差和棋盘格重投影误差
├── result_visualizer.py    # 标定结果可视化
├── calibration/
│   ├── solver_axxb.py      # AX=XB SVD 求解
│   ├── optimizer.py        # 非线性优化
│   ├── transforms.py       # 坐标变换工具
│   └── feature_extractor.py # 棋盘格角点检测
├── camera/realsense.py     # RealSense D405 驱动
├── robot/ur_robot.py       # UR3 通信
├── data/                   # 标定数据
├── results/                # 标定结果
└── tests/                  # 验证程序
```

## 安装

```bash
pip install -r requirements.txt
```

## 标定模式

| 模式 | 相机安装方式 | 标定结果 `X` 的含义 | 常用变换 |
| --- | --- | --- | --- |
| `eye_on_hand` | 相机安装在机械臂末端 | `T_tcp_camera` | `T_base_target = T_base_tcp @ X @ T_camera_target` |
| `eye_to_hand` | 相机固定在工作空间 | `T_base_camera` | `T_base_target = X @ T_camera_target` |

这里的 `T_camera_target` 是相机观测到的标定目标位姿。棋盘格后端中，目标是棋盘格；AprilTag 后端中，目标是指定 tag。

## 当前主流程

1. 运行 `python main.py`。
2. 选择标定模式：`eye_on_hand` 或 `eye_to_hand`。
3. 选择观测后端：棋盘格或 AprilTag。
4. 选择新采集数据，或使用 `data/{mode}` 下已有数据。
5. 每帧数据保存机械臂 TCP 位姿和相机观测到的目标位姿。
6. 使用所有帧两两组合构造 AX=XB 相对运动方程，通过 SVD 求解初值。
7. 以 SVD 结果为初值，对 6 自由度手眼矩阵做非线性优化。
8. 保存结果并计算误差。

优化阶段只优化手眼矩阵 `X`，不再优化深度缩放。`z_scale` 固定为 `1.0`，结果目录中的 `depth_scale.txt` 只是为了兼容已有结果读取逻辑。

## 观测后端

### 棋盘格

棋盘格角点由 `calibration/feature_extractor.py` 检测，位姿由 `solvePnP` 根据棋盘格几何尺寸和相机内参估计。

相关配置在 `CHECKERBOARD_CONFIG`：

```python
CHECKERBOARD_CONFIG = {
    'size': (11, 8),
    'square_size': 0.006,
    'board_to_base_rough': [...],
    'board_to_tcp_rough': [...],
}
```

`board_to_base_rough` 和 `board_to_tcp_rough` 只用于误差评估和可视化，不参与手眼矩阵求解。

### AprilTag

AprilTag 后端根据 tag 检测结果和 tag 尺寸估计 `T_camera_tag`。

相关配置在 `APRILTAG_CONFIG`：

```python
APRILTAG_CONFIG = {
    'family': 'tag36h11',
    'tag_size': 0.03,
    'target_tag_id': 1,
    'decision_margin_threshold': 20.0,
    'min_area_ratio': 0.0005,
}
```

AprilTag 后端当前跳过棋盘格重投影误差，只计算位姿参考误差和旋转参考误差。

## 误差含义

- **位置参考误差**：把每帧观测到的目标位姿通过手眼矩阵和机器人位姿转换到参考坐标系后，与粗略配置位姿比较。这个误差依赖 `board_to_base_rough` 或 `board_to_tcp_rough` 的准确性，因此主要用于参考。
- **旋转参考误差**：与位置参考误差类似，但比较旋转角度差，单位为 degree。
- **重投影误差**：仅棋盘格后端使用。根据求解出的位姿把棋盘格角点重新投影到图像上，并与检测到的角点比较，单位为 pixel。

如果参考位姿本身不准，位置参考误差可能偏大；这不一定代表手眼矩阵完全错误。更可靠的判断方式是结合重投影误差、每帧残差分布，以及实际机器人验证程序。

## 输出文件

结果保存在 `results/{mode}/`：

```text
results/{mode}/
├── handeye_transform.txt  # 4x4 手眼矩阵 X
├── depth_scale.txt        # 固定为 1.0，仅兼容保留
└── calibration_info.txt   # 模式、z_scale 和矩阵文本
```

实际使用时优先读取 `handeye_transform.txt`。

## 数据建议

- 至少采集 6 帧，建议采集更多帧。
- 机械臂姿态变化要充分，尤其要有明显的旋转变化。
- 避免所有采样点集中在很小范围内。
- 棋盘格或 tag 应尽量覆盖不同图像区域和不同距离。
- 如果某几帧检测质量明显差，建议删除后重新求解。

## 验证程序

`tests/` 下的脚本用于实际验证标定结果，不属于主标定流程。其中 `tests/move_tcp_eye_to_hand_apriltag.py` 会读取结果矩阵并结合 AprilTag 观测做运动验证。

## 可选研究方向

- **RGB-D 重投影优化**：将棋盘格角点像素结合 RealSense 深度图恢复为相机坐标系下的 3D 点，再通过手眼矩阵和机器人位姿建立重投影残差进行非线性优化。当前主流程未采用。
- **RGB-D 棋盘格位姿估计**：直接用角点像素和深度图恢复棋盘格 3D 角点，再拟合棋盘格平面与坐标系，作为 `solvePnP` 之外的观测方式。当前主流程未采用。
