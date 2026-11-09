# R2 Hand-Eye Calibration

采集机器人与相机数据，完成手眼标定、静态观测一致性分析和标定结果评价。支持 UR、法奥 FR3 状态采集与 RealSense 相机。

## Part 1：快速开始

### 环境配置

在项目根目录执行；硬件依赖需在实际采集机器上安装。

```powershell
conda create -n r2-handeye python=3.12 -y
conda activate r2-handeye
python -m pip install -r requirements.txt
```

纯离线计算使用 NumPy、SciPy、OpenCV、Matplotlib，不初始化相机或机器人。此次重构未重新安装依赖。

FR3 控制器 V3.8.6 使用 [法奥 Python SDK v2.1.6](https://github.com/FAIR-INNOVATION/fairino-python-sdk/releases/tag/v2.1.6_robot3.8.6)。将 SDK 解压到 `third_party/fairino-python-sdk-2.1.6_robot3.8.6/`，保持内部目录结构。驱动直接加载 `Robot.py`，无需编译或配置 `PYTHONPATH`。`configs/robots/fr3_lab.json` 中的 `sdk_path` 指向包含 `fairino/Robot.py` 的目录，默认使用 `windows`；Linux 改为 SDK 的 `linux` 目录。`tcp_host_ip` 默认填入出厂地址，使用前改为实际控制器 IP。

### 项目运行

全部使用命令行参数，不设终端选择菜单。先检查所选工作流配置引用的设备和标定板文件。

UR3 和 FR3 共用 `configs/capture.json`。使用 FR3 时，将其中的 `robot_config` 改为：

```json
"robot_config": "configs/robots/fr3_lab.json"
```

使用 UR3 时设为 `configs/robots/ur3_lab.json`，采集命令相同：

```powershell
python -m scripts.collect --mode eye_to_hand --config configs/capture.json
```

眼在手上改为 `--mode eye_on_hand`。机器人静止后，Space 检测并记录图像和机器人状态快照，Enter 保存该快照，Esc 结束。新采集继续要求实际 TCP 六维偏置全零；程序读取核验，不修改控制器 TCP。

将下面的示例路径替换为采集输出的实际目录：

```powershell
$data = "data/eye_to_hand/20261104_143025"
python -m scripts.calibrate --data "$data" --solver-config configs/solver.json
```

标定完成后自动计算误差并保存图表；增加 `--visualize` 才显示图表窗口。`calibrate` 与 `analyze calibration` 共用评价入口，默认读取 `configs/references/lab_board_mount.json`；可用 `--reference-config` 指定其他参考配置：

```powershell
python -m scripts.calibrate --data "$data" --reference-config configs/references/lab_board_mount.json
```

静态观测分析不需要机器人或手眼结果。保持相机和标定板相对固定，Enter 保存一次观测，Space 或 Esc 结束采集并分析。至少保存两帧；不足时原始数据仍保留。

```powershell
python -m scripts.analyze observation --capture --config configs/observation.json
python -m scripts.analyze observation --data data/observation/20261104_150000
```

手眼结果分析复用已求出的矩阵，不重新求解；默认使用该结果记录的原标定数据。指定另一批数据时，使用该批数据评价空间一致性和重投影误差。

```powershell
$result = "outputs/eye_to_hand/20261104_144000"
python -m scripts.analyze calibration --result "$result"
python -m scripts.analyze calibration --result "$result" --data data/eye_to_hand/20261104_160000
```

独立矩阵文件没有来源信息，必须同时指定 `--data` 和 `--mode`。两个分析模式均支持 `--visualize`、`--visualization-config` 和 `--output-root`；观测分析的 `--config` 仅用于新采集。

运动验证沿用 UR RTDE 和单 Tag 流程，先试算：

```powershell
python -m scripts.move_tcp --result "$result" --config configs/motion.json --dry-run
```

`--dry-run` 不发送运动指令；眼在手上仍需读取机器人当前位姿。去掉该参数且配置中 `dry_run` 为 `false` 时，Enter 锁定目标并发送运动，c 清除显示中的锁定目标，Esc 退出程序。退出或清除显示不等同于机器人停止命令。到位指标比较机器人回报的 TCP 与计算目标，不能作为独立绝对精度证明。

无硬件回归测试：

```powershell
python -X utf8 -B -m unittest discover -s tests -p "test_*.py" -v
```

## Part 2：输入与输出

### Input：配置读取关系

所有相对路径均相对项目根目录。修改设备参数只改被引用的设备文件，不在工作流中重复填写。

| 操作 | 读取配置与数据 |
|---|---|
| `collect` | `capture.json` → 所引用的机器人、相机、标定板和检测配置 |
| `calibrate` | 已保存的数据及采集信息＋`solver.json`＋绘图配置；可选参考配置 |
| `analyze observation --capture` | `observation.json` → 相机、标定板、检测配置；分析时读取绘图配置 |
| `analyze observation --data` | 已保存的观测数据＋绘图配置，不读取当前设备或检测配置 |
| `analyze calibration` | 指定结果和数据＋绘图配置；可选参考配置，不读取求解或设备配置 |
| `move_tcp` | `motion.json` → 机器人、相机、单 Tag 和检测配置；指定手眼结果、绘图配置 |

| 文件 | 参数职责 |
|---|---|
| `configs/robots/ur3_lab.json` | 驱动、IP、通信与只读状态端口、工作空间及驱动默认关节速度/加速度 |
| `configs/robots/fr3_lab.json` | 法奥驱动、实际控制器 IP、SDK 源码目录和新状态等待超时（秒） |
| `configs/cameras/d405_lab.json` | 驱动、序列号、分辨率、帧率、`enable_depth`；RealSense 内参来自设备 |
| `configs/targets/*.json` | 板类型、实际尺寸、Tag ID、布局及旋转；不含安装参考位姿 |
| `configs/detection.json` | `checkerboard`、`apriltag_board`、`apriltag_single` 检测配置组 |
| `configs/capture.json` | 设备、板及检测配置引用；采图前后机器人静止阈值 |
| `configs/observation.json` | 相机、板及检测配置引用；两次保存的最短间隔（秒） |
| `configs/motion.json` | 配置引用、目标 TCP 相对 Tag 的位姿、运动速度/加速度、到位阈值 |
| `configs/solver.json` | 最少样本、最小二乘、位置/旋转权重与归一化尺度 |
| `configs/references/lab_board_mount.json` | 按目标类型记录参考位姿、单位、TCP 定义及测量来源 |
| `configs/visualization.json` | 图表尺寸与运动预览坐标轴长度 |

机器人配置的 `default_velocity/default_acceleration` 作用于 UR 驱动的关节运动默认值；`move_tcp` 使用 `motion.json` 的 `rtde_velocity/rtde_acceleration`（rad/s、rad/s²）。采集不会主动运动机器人。工作空间参数供 UR 驱动工作空间查询使用，不代表 RTDE 路径规划限制。

FR3 驱动只读取状态，不发送运动或修改工具坐标系。每次采样等待新的反馈包，从同一快照读取法兰位姿和实际工具偏置；偏置六个分量须全零，此时法兰即采集所用 TCP。法奥的毫米转换为米，固定轴 XYZ RPY（度）转换为旋转矩阵；反馈不更新或断线时拒绝采样并报错，SDK 重连循环由适配器关闭，需重新启动采集。FR3 支持两种手眼模式的采集及后续离线计算；`move_tcp` 运动验证仍仅支持 UR。首次实机使用时需对照示教器核对基座系位姿和方向。

换棋盘格或另一块 Tag 板时，修改工作流的 `target_config` 和匹配的 `detection_profile`。静态观测支持棋盘格、整板和单 Tag；手眼采集保留棋盘格与整板，运动验证目前使用单 Tag。

D405 彩色流的非零 inverse_brown_conrady 在相机驱动中按 RealSense SDK 模型校正；彩色图和已对齐深度使用同一映射，深度单位仍为 m。输出图像及保存的 image.png 对应校正后内参与零畸变，设备原始模型、系数和内参另存于 session.json 的 source_camera_model。其他已支持模型保持原有处理。

采集会保存实际内参、图像尺寸、板几何和检测设置。离线处理以保存内容为准，不因修改当前配置而改变历史数据解释，也不自动重新检测图像。`null` 表示未提供；例如参考位姿为 `null` 时跳过对应比较。

位姿统一约定为 `T_A_B`：将 B 系坐标转换到 A 系。平移使用米，内部旋转向量使用弧度，不是欧拉角；参考文件明确声明 m/mm、deg/rad，按声明转换。

| 标定模式 | 输入机器人位姿 | 输出手眼矩阵 | 板应固定在哪个坐标系 |
|---|---|---|---|
| `eye_to_hand` | `T_base_tcp` | `T_base_camera` | TCP |
| `eye_on_hand` | `T_base_tcp` | `T_tcp_camera` | 基座 |

新采集 TCP 偏置为零，因此 TCP 与法兰重合。固定安装和 TCP 定义必须与待评价结果相符；程序无法仅凭文件证明安装未改变。

### Output：目录与文件

```text
data/
├── eye_to_hand/<data_id>/
├── eye_on_hand/<data_id>/
└── observation/<data_id>/
    ├── session.json
    └── frames/000001/
        ├── image.png
        ├── depth.npy          # 可选：有深度才保存，单位 m
        ├── robot.json         # 仅手眼数据：位姿、前后状态、TCP 偏置
        ├── observation.json   # 位姿、2D/3D 对应点、投影模型、质量
        └── frame.json         # 时间、文件关联及完成标记

outputs/
├── eye_to_hand/<calibration_id>/
│   ├── result.json
│   ├── handeye_transform.txt
│   ├── evaluation.json
│   ├── report.html
│   ├── residuals.csv
│   └── figures/
├── eye_on_hand/<calibration_id>/
├── analysis/<analysis_id>/
│   ├── run.json
│   ├── analysis.json
│   ├── evaluation.json
│   ├── report.html
│   ├── residuals.csv
│   └── figures/
└── motion_validation/<motion_id>/
    ├── run.json
    ├── camera.json
    └── events.json
```

手眼标定与 `analyze calibration` 均生成五张图：`pose_errors.png`（相对均值的一致性残差）、`components.png`（一致性分量）、`reprojection.png`（观测与 chain 重投影）、`reference_pose_errors.png`（相对配置参考位姿的偏差）、`reference_components.png`（参考板坐标系下的偏差分量）。不再生成 `spatial.png`。chain 仍使用样本平均安装位姿。参考字段为眼在手外的 `board_to_tcp_rough` 或眼在手上的 `board_to_base_rough`；单位和旋转表示以参考配置声明为准，参考值缺失时对应图明确标注无数据。

`handeye_transform.txt` 的矩阵固定保存 8 位小数，`result.json` 中同一矩阵同步舍入到 8 位小数；求解内部保留原精度，评价使用导出矩阵以保证标定与后续分析一致。

三个数据类别使用相同帧目录结构，观测数据没有机器人文件。原图可以是彩色或灰度，检测绘制内容不覆盖原图。RealSense 深度可关闭；开启时对齐到彩色图并换算为米，**当前手眼算法不使用深度修正**。

每次采集、求解和独立分析新建目录，以本机时间命名，同秒重名增加序号；不覆盖旧文件。帧完成标记最后写入，不完整帧不参与读取。`--data-root` 指定数据分类目录的父目录；`--output-root` 指定本次新结果目录的父目录。

`calibrate` 的附带评价保存在本次标定目录，显式 `analyze` 保存在新的分析目录。两者共用同一手眼评价函数，相同矩阵、数据和评价设置应得到相同指标。报告使用 mm、deg、px；逐帧数据见 CSV。每次评价另生成 report.html，汇总矩阵、指标和图表；支持放大图表、展开逐帧明细和下载数值。图片及 JSON/CSV 内嵌，可单独复制 HTML 离线查看；生成时核对汇总统计与逐帧数据一致。图表保存位置/旋转残差、分量及重投影曲线。

JSON 保留必要的来源路径、配置和格式版本，不计算文件哈希；短数组单行，矩阵逐行。来源文件关联依靠路径，请保留原始数据；矩阵仍与同目录 `result.json` 中的值核对。

旧版 `session.json`＋`poses/images` 会话可直接读取，不迁移或改写。旧静态验证 CSV 可传给 `analyze observation --data`，其重投影只能使用已有列值，因为 CSV 没有完整原始角点。

缺少会话信息的旧整板位姿数据可显式导入，只评价位姿一致性：

```powershell
python -m scripts.calibrate --data data/old_poses --legacy-poses --mode eye_to_hand --backend four_apriltag
```

## Part 3：算法流程

```text
collect：图像＋采图前后机器人状态 → 零 TCP/静止检查 → 板检测 → 保存数据
calibrate：读取配对位姿 → 运动约束检查 → SVD 初值 → 非线性最小二乘 → 共用评价
analyze observation：静态采集或读取 → 相机系位姿相对均值的波动＋观测重投影
analyze calibration：读取矩阵及配对数据 → 共用评价（不重新求解）
```

保留原有求解算法和默认数值：至少 6 帧，优化使用位置/旋转归一化残差，优化失败或变差时保留 SVD。棋盘格使用原图角点；整板使用 alpha=0 去畸变图像，由 pyapriltags 直接估计各 Tag 位姿，按 T_camera_board = T_camera_tag × inverse(T_board_tag) 换算后，沿用提交 7d9d24c 的 decision_margin 加权平移均值与 SO(3) 旋转均值；不调用 OpenCV PnP，重投影仅用于质量检查；单 Tag 观测与运动验证沿用 alpha=1 去畸变与 Tag 位姿估计。新数据保存对应投影模型，读取时检查一致性。

| 指标 | 含义 |
|---|---|
| 静态观测重复性 | 相机系位置波动及相对平均姿态的旋转波动，不证明手眼精度 |
| 空间一致性 | 经手眼链路计算的板位姿相对平均位姿的离散 |
| 观测重投影 | 视觉位姿预测的角点与检测角点之差 |
| 全链路重投影 | 机器人、手眼矩阵和该批数据的平均板位姿共同预测的角点偏差 |
| 参考位姿偏差 | 与显式提供的参考比较，包含参考测量自身误差 |

原标定数据报告为拟合评价；另采数据报告为另一数据集的一致性评价。两者的全链路重投影均使用本批数据估计平均板位姿，不等同于有外部真值的独立绝对精度测试。

## Part 4：工程结构

```text
configs/               # 工作流配置及 robots/cameras/targets/references
scripts/               # collect、calibrate、analyze、move_tcp 四个命令入口
handeye/
├── robot/             # 状态接口、UR/法奥采集驱动、独立 UR RTDE 运动能力
├── camera/            # 相机接口、RealSense 驱动及畸变模型
├── targets/           # 板几何和共用检测器
├── calibration/       # 坐标变换、SVD、优化；保留旧几何导入兼容
├── analysis/          # observation、calibration 与共用报告输出
├── config.py          # 按需加载配置，解析项目相对路径
├── session.py         # 独立目录、JSON 与帧读写、结果选择
├── pipeline.py        # 采集和求解流程
├── data_collector.py  # 采集快照、静止检查、新旧数据读取
├── device_manager.py  # 按驱动类型创建设备
├── calibration_solver.py
├── error_calculator.py
├── move_validation.py
└── result_visualizer.py
data/                  # 原始输入
outputs/               # 求解、分析和运动验证结果
tests/                 # 不连接硬件的回归测试
third_party/           # 本地下载的法奥 SDK（不纳入版本控制）
```

`validate` 已合并进 `analyze observation`，旧 `analyze --kind base/consistency` 替换为两个明确的子命令。新增设备时实现接口并登记驱动工厂；目前未实现工业相机或多相机。
