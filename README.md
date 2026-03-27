# 手眼标定系统

基于 UR3 机械臂和 Intel RealSense D405 相机的手眼标定工具。

## 设备

- **机械臂**: UR3 (TCP/IP, 端口 30003)
- **相机**: Intel RealSense D405

## 项目结构

```
handeye/
├── main.py                 # 主入口程序
├── config.py               # 配置文件
├── device_manager.py       # 设备连接管理
├── data_collector.py       # 数据采集模块
├── calibration_solver.py   # AX=XB 标定求解
├── error_calculator.py     # 误差计算
├── result_visualizer.py    # 结果可视化
├── calibration/            # 底层算法
|   ├── svd.py             # SVD算法
│   ├── solver_axxb.py     # SVD求解器
│   ├── optimizer.py       # 非线性优化
|   ├── transforms.py    # 坐标变换工具
│   └── feature_extractor.py # 棋盘格角点检测
├── robot/                  # 机械臂驱动
│   └── ur_robot.py        # UR3通信
├── camera/                 # 相机驱动
│   └── realsense.py       # D405驱动
├── data/                   # 标定数据存储
└── results/                # 标定结果存储
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置 (config.py)

```python
# 机械臂IP
UR3_CONFIG = {'tcp_host_ip': '192.168.56.102', 'tcp_port': 30003}

# 棋盘格参数
CHECKERBOARD_CONFIG = {
    'size': (11, 8),        # 内角点数量
    'square_size': 0.006,   # 方格大小(米)
    'board_to_base_rough': [x, y, z, rx, ry, rz],  # Eye-on-Hand粗略位姿
    'board_to_tcp_rough': [x, y, z, rx, ry, rz],   # Eye-to-Hand粗略位姿
}

# AprilTag参数
APRILTAG_CONFIG = {
    'family': 'tag36h11',
    'tag_size': 0.03,  # 米
    'target_tag_id': 1,
    'decision_margin_threshold': 20.0,
    'min_area_ratio': 0.0005,
}
```

## 使用方法

### 运行程序

```bash
python main.py
```

### 操作流程

1. **选择模式**: 输入 `1` (Eye-on-Hand) 或 `2` (Eye-to-Hand)
2. **选择标定板**：输入 `1` (棋盘格) 或 `2` (AprilTag)
3. **自动连接**: 连接 UR3 和 D405
4. **数据采集**:
   - 人工移动机械臂到新位置（示教器人工示教）
   - 按 `Space` 检测角点并显示结果
   - 按 `Enter` 保存当前有效检测帧和TCP位姿
   - 按 `Esc` 退出采集
   - 至少采集 6 帧（建议更多）
5. **自动计算**: 基于 AX=XB 的 SVD 求解
6. **误差报告**: 显示位置参考误差 + 全角点重投影误差
7. **可视化**: 3D显示坐标系，误差分布图

## 标定模式

| 模式        | 说明       | 求解目标       |
| ----------- | ---------- | -------------- |
| Eye-on-Hand | 相机在末端 | 相机→TCP 变换  |
| Eye-to-Hand | 相机固定   | 相机→基座 变换 |

## 粗略位姿用途

- 仅用于**可视化参考**显示标定板位置
- 仅用于**误差计算**作为参考对比
- **不参与**标定求解

## 数据输出

```
results/{mode}/
├── handeye_transform.txt  # 手眼变换矩阵 (4x4)
├── depth_scale.txt       # 深度缩放因子
└── calibration_info.txt  # 详细信息
```

## 误差指标

- **位置参考误差**: 与粗略先验位姿对比（仅参考）
- **旋转参考误差**: 与粗略先验位姿对比（仅参考）
- **重投影误差**: 基于棋盘格全部角点的像素残差统计

## SVD特征实验 (data/svd)

新增了基于 RealSense 的 RGB-D 保存与离线 SVD 特征提取流程，所有实验数据统一保存到 `data/svd`。

### 1) 采集RGB-D帧

```bash
python tests/capture_svd_data.py
```

按键说明:

- `Space`: 保存当前一帧到 `data/svd/images`
- `Q` 或 `Esc`: 退出

输出文件:

- `data/svd/images/rgb_XXX.png`
- `data/svd/images/depth_XXX.npy`
- `data/svd/images/timestamps.csv`
- `data/svd/images/camera_intrinsics.json`

### 2) 离线SVD特征提取

```bash
python -m calibration.svd
```

输出文件:

- `data/svd/svd_features.csv`
