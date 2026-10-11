# R2 Hand-Eye Calibration

UR3 与 RealSense 的手眼标定工具，支持棋盘格、单 AprilTag 和多 AprilTag 刚性板。
提供实时采集、离线复算、空间一致性和重投影评价。标定模式为眼在手上与眼在手外。

## 运行

使用 Python 3.10+，安装 `requirements.txt` 中的依赖后，在项目根目录运行：

```powershell
python main.py
```

依次选择手眼模式、标定板类型，以及新采集或离线复算。
离线复算不连接机器人和相机，也不要求安装 RealSense SDK 或 AprilTag 检测器；
仍需 NumPy、SciPy、OpenCV 和 Matplotlib。

采集按键：空格检测并记录图像/TCP 快照，Enter 保存该快照，Esc 结束。
必须在机器人静止时采集。采集前后 TCP 变化超过 0.5 mm 或 0.1° 时拒绝样本，
阈值位于 `CALIBRATION_CONFIG`。这是静止采集检查，不是硬件同步。

## 坐标与单位

`T_A_B` 将 B 系坐标转换到 A 系。平移单位米，内部旋转向量单位弧度。
粗略参考位姿配置的旋转向量沿用角度输入，`tag_rotvecs` 则使用弧度。
TCP 指机器人当前配置的实际 TCP，不应直接等同于法兰；采集期间不得更改 TCP 设置。

| 模式 | 手眼结果 X | 应固定的目标位姿 |
|---|---|---|
| `eye_on_hand` | `T_tcp_camera` | `T_base_tcp @ X @ T_camera_target` |
| `eye_to_hand` | `T_base_camera` | `inverse(T_base_tcp) @ X @ T_camera_target` |

## 模块职责

| 模块 | 职责 |
|---|---|
| `main.py` | 交互、会话选择和评价流程；设备始终在 finally 中释放 |
| `data_collector.py` | 静止采集快照、检测预览和样本保存/读取 |
| `calibration/session.py` | 会话清单、独立运行目录、原子 JSON 写入 |
| `calibration/targets.py` | 棋盘格/Tag 几何、固定角点顺序、多标签联合 PnP |
| `calibration/transforms.py` | 刚体变换检查、稳定旋转转换、旋转均值和固定系目标位姿 |
| `calibration/solver_axxb.py` | 运动退化检查、AX=XB 的 SVD 初值 |
| `calibration/optimizer.py` | 无量纲、向量化的最小二乘优化及接受检查 |
| `calibration_solver.py` | 数据加载、求解编排和结果保存 |
| `camera/model.py` | 相机模型检查、按分辨率缓存去畸变映射 |
| `error_calculator.py` | 空间一致性、参考偏差、固定对应关系的重投影评价 |

旋转运算统一使用 SciPy Rotation，避免 180° 奇异问题。
SVD 使用精简矩阵分解，相对位姿逆矩阵预先计算；优化批量计算残差。
离线加载不读取未参与求解的深度数组。去畸变映射只在尺寸变化时重建。

## 标定板与相机配置

棋盘格的 `size` 表示内角点列数和行数，`square_size` 为实测格长。
单 AprilTag 使用 `pyapriltags` 的位姿估计，并检查正深度及有序角点重投影误差。
标签尺寸是检测角点之间的边长，不包括外围留白。

多标签板由 `tag_ids`、`tag_centers`、`tag_rotvecs` 描述。
每个标签的中心及旋转均相对于板坐标系；旋转向量由解码后的 Tag 坐标系变换到板坐标系。
默认所有标签同向，若实际旋转了标签，必须填写相应旋转。
所有可见角点联合求板位姿，不再按解码置信度平均独立位姿。
默认至少可见两个标签，每个标签平均重投影误差不超过 2 px；可在配置中调整。
任一标签几何不一致、重复 ID 或角点数据异常时拒绝该帧。

单标签与棋盘格的粗略参考位姿分别配置，不再交叉引用。
没有独立参考时设为 `None`，仅评价空间一致性。

相机宽高、帧率、设备 ID 显式传入驱动；会话记录实际启用的分辨率和序列号。
非零的 inverse/modified Brown-Conrady 等模型不会直接当作 OpenCV 畸变使用，
遇到不兼容模型会明确报错。当前实机模型需要连接相机后确认。

## 会话与结果

新采集保存在 `data/{mode}/sessions/{id}/`，不会删除历史数据。

```text
session.json            # 模式、板几何、K/D、相机标识、图像尺寸、角点坐标系
poses/tcp_001.txt        # 取图时记录的实际 TCP
poses/tag_pose_001.txt   # 或棋盘格 corners_001.txt
poses/tag_corners_001.txt
poses/tag_ids_001.txt
poses/frame_001.json     # 最后写入的完成标记、取图时间、采集耗时
images/rgb_001.png       # 原始图像
images/depth_001.npy     # 深度存档，不参与当前手眼求解
```

棋盘格角点属于原图；AprilTag 角点属于 alpha=0、相同输出尺寸的去畸变图。
采集、保存和报告使用相同的坐标约定。写入中断而没有完成标记的样本不会加载。
离线复算使用会话记录的参数；修改 `config.py` 不会改变历史样本的含义。
不要直接编辑历史会话的几何或内参来复算已保存的 Tag 位姿；应重新采集或重新检测原图。

旧数据没有 `session.json` 时，AprilTag 允许显式选择“历史位姿一致性”模式，
不报告无法追溯的重投影误差，不允许追加采集。旧棋盘格数据请重新采集为完整会话。

每次求解保存在 `results/{mode}/runs/{id}/`：

- `handeye_transform.txt`：手眼矩阵。
- `calibration_info.txt`、`depth_scale.txt`：保留文本格式；scale 固定为 1。
- `result.json`：来源会话、板配置、求解诊断及优化是否被接受。
- `residuals.csv`：按原始帧编号对齐的一致性和全链路重投影误差。

历史根目录下的结果文件保留，不自动覆盖。实际验证时显式指定本次结果，例如：

```powershell
python tests/test_eye_to_hand_tag36h11_id0_pose_consistency.py --handeye-file results/eye_to_hand/runs/<id>/handeye_transform.txt --csv tests/new_consistency.csv
python tests/analyze_base_tag_from_calibration.py --calibration-info results/eye_to_hand/runs/<id>/calibration_info.txt --csv tests/new_consistency.csv --output tests/new_base_analysis.png
```

运动验证脚本 `tests/move_tcp_eye_to_hand_apriltag.py` 也支持 `--handeye-file`。
该脚本会发送机器人运动指令；初次验证使用其 `--dry-run` 参数。

## 优化与误差解释

仅有平移或几乎绕单轴转动的样本会被拒绝；建议采集至少 15–25 个覆盖不同旋转轴、
图像区域和距离的静止姿态。最低数量仍为 6 帧，数量足够不代表约束足够。

优化残差按位置尺度 1 mm、旋转尺度 1° 归一化，再应用位置/旋转权重。
这两个尺度是可配置的工程权重，并非测得的相机噪声。
最小二乘未收敛、输出非有限值或同一目标函数变差时，保留 SVD 初值。
界面同时显示 SVD 与最终结果的位置/旋转均值；位置改善可能伴随旋转误差上升。

- 空间一致性和全链路重投影为同批标定样本的拟合指标，不是独立绝对精度。
- 相对手工粗略位姿的偏差包含参考值自身的误差，单独标注。
- AprilTag 重投影按固定 ID 和角点顺序计算，不允许重排角点来降低误差。
- 空评价显示 N/A，不再显示零误差。
- 静态 Tag 重复测量衡量视觉稳定性；相对平均位姿的残差会抵消固定手眼矩阵，不能验证该矩阵的准确度。
- `analyze_base_tag_from_calibration.py` 的平移分量现在在基座系计算，Base-Z 名称与计算一致；旋转残差仍相对平均 Tag 姿态。

手眼验证应补采未参与求解的机器人姿态，检查固定目标的一致性；绝对精度还需要独立测量参考。

## 自动化验证

```powershell
python -B -m unittest discover -s tests -p test_calibration_core.py -v
```

覆盖两种手眼模式、旋转奇异角、退化运动、优化回退、TCP 拆包/粘包、快照匹配、
采集中运动、写入失败、历史数据保护、板上标签旋转、错误角点对应、相机模型及完整离线流程。
这些测试不连接机器人，不代替 RealSense 与 UR 实机验收。
