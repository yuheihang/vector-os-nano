# 项目核心基准文档

## 一、文档基本信息

- **项目名称**：Q-GRASP：面向 Go2+ARX 狗臂机器人的 Agent 长程导航抓取系统（暂定工作名）
- **Q-GRASP 暂定全称**：Quadruped-arm Grasp-aware Reasoning and Attention-based Scene Planning
- **文档版本**：V1.0（基准版）
- **生成日期**：2026-07-15
- **生成依据**：当前对话全量历史、当前 `vector-os-nano` 仓库 README 与关键代码、用户提供的研究框图及 UniLM-Nav 论文
- **基础项目路径**：`/media/fishyu/fish-14tb-2/YuXi/vector-os-nano`
- **文档定位**：本文件用于固化当前已确认的研究目标、系统边界、三项核心创新、实现路线和待确认问题。后续方案设计、代码实现和论文讨论均应以本文档为基准；任何改变必须显式记录版本更新。

> 状态说明：本文中的“已确认”表示用户已经明确提出，或已由当前仓库代码核实；“当前建议”表示对话中已经提出但用户尚未逐项明确确认的工程建议；“待确认”表示对话中仍存在歧义或尚未作出最终决策。候选名称、模块名称和论文标题不等同于最终命名。

### 1.1 核心术语

| 术语 | 本文含义 |
|---|---|
| Agent | 接收自然语言、拆解长程任务、选择 Skill、监控结果并决定后续动作的高层智能体，不直接输出关节控制量 |
| Skill | 面向 Agent 暴露的结构化机器人能力，如目标检测、导航、最佳视点选择、联合抓取规划、抓取执行和结果验证 |
| Service | 被 Skill 调用的可复用算法或外部能力，如 YOLO、SAM、SysNav、GraspNet、AnyGrasp、MLLM、IK、碰撞检测和 Scene Graph 存储 |
| Pair | 一组绑定的 `(base pose, grasp pose)` 候选，表示一套完整的狗臂移动抓取方案 |
| `b_view` | 为获得清晰目标观测而选择的最佳观察基座位姿 |
| `b_grasp` | Pair Attention Planner 最终选择、用于实际抓取的基座位姿 |
| Context Tokens | 目标物体、机器人、场景和任务上下文经过编码后的网络表示 |
| Pair Tokens | 每组 base-grasp Pair 及其执行前可获得特征经过编码后的网络表示 |

## 二、项目背景与核心目标

### 2.1 研究背景

当前研究基于开源项目 **Vector OS Nano** 开展。该项目的目标是让 Agent 通过自然语言控制不同机器人，并通过已注册的 Skill 或 Tool 完成长链任务。当前仓库已经具备以下基础结构：

- `vector-cli`：自然语言 Agent 交互入口；
- `vector-os-mcp`：通过 MCP 暴露机器人 Skill 和世界状态资源；
- VGG（Verified Goal Graph）/Agent 层：将自然语言目标拆解为可验证的子计划；
- `@skill`、`SkillRegistry` 和 `SkillContext`：Skill 注册、发现和执行机制；
- `BaseProtocol`、`ArmProtocol`：与具体机器人解耦的移动底盘和机械臂接口；
- `WorldModel` 和 `SceneGraph`：共享世界状态与空间记忆；
- MuJoCo、Go2 ROS2 proxy、导航桥和演示级机械臂抓取能力。

本项目原有能力可以概括为：

1. 单机械狗导航：Go2 可通过导航 Skill、ROS2 proxy 和导航栈进行房间或坐标导航；
2. 单机械臂控制与演示级抓取：现有 `pick_top_down` 依赖已知物体位姿，主要执行顶部抓取；
3. 简单移动抓取：现有 `mobile_pick` 本质上是“导航到接近位置，再调用顶部抓取”的顺序组合，尚未实现 base 与 grasp 的联合决策。

### 2.2 用户已有基础

用户已经通过其他项目训练了 Go2+ARX 狗臂机器人的全身协同底层策略，并确认具备以下能力：

- Go2 的 12 个腿部自由度由已有策略控制；
- 机械臂通过 IK 控制；
- 底层策略能够进行位姿跟踪；
- 已表现出狗和机械臂全身协同能力。

因此，本项目不以从零训练基本行走或机械臂 IK 为主要目标，而是重点研究：如何让 Agent 通过结构化 Skill，将导航、观察、联合站位与抓取规划、全身执行和长期记忆组织成完整系统。

### 2.3 核心痛点

现有导航与抓取模块机械串联时存在以下根本问题：

1. **导航终点与抓取需求不一致**：导航系统通常只负责到达目标附近，不能保证最终位置适合观察或抓取。
2. **base pose 与 grasp pose 分别优化**：单独最优的站位和抓取位姿组合后可能不可达、碰撞或不适合狗臂全身策略执行。
3. **缺少长期抓取记忆**：重复抓取同一物体或同类物体时，Agent 无法复用历史最佳视点、成功 Pair 和失败经验。
4. **长程任务缺少闭环组织**：单次调用导航和抓取能力不足以稳定完成“寻找—导航—观察—抓取—验证—运输—放置”等长序列任务。
5. **当前硬件形态不一致**：基础项目中的 Go2+机械臂仿真主要围绕 Piper，实验室实际平台为 Go2+ARX，需要完成 embodiment adapter、坐标系、控制接口和传感器适配。

### 2.4 核心目标

构建一个面向 Go2+ARX 狗臂机器人的自然语言 Agent 系统，使其能够：

- 根据自然语言拆解并执行长程任务；
- 调用导航、感知、最佳视点选择、抓取规划、抓取执行和验证等 Skill；
- 将 GraspNet/AnyGrasp 等抓取候选与 GORM-like base 候选组成 Pair；
- 通过 Pairwise Base-Grasp Attention Planner 联合选择适合全身执行的站位与抓取位姿；
- 通过任务触发式 Scene Graph 保存目标物体视点、Pair 和执行历史；
- 将结构化目标交给已有全身策略、ARX IK 和夹爪执行；
- 在执行反馈基础上进行验证、失败恢复和后续规划；
- 形成可用于 SCI 论文的完整方法、系统、实验和消融结果。

### 2.5 项目边界与非目标

#### 已确认边界

- Agent 负责语义任务规划和 Skill 编排，不直接控制 Go2 腿关节或 ARX 关节。
- 底层全身策略是执行层，本研究重点不是重新发明基础腿部运动控制。
- 第一版联合规划器不对多个 SE(2)/SE(3) 姿态做加权平均，而是选择一个完整 Pair。
- 第一版不训练姿态残差网络；选中 Pair 后优先使用 IK、碰撞检测或局部几何搜索做修正。
- Scene Graph 不为所有场景物体存储稠密抓取信息，只对实际抓取目标按需附加操作记忆。
- `reachability`、`stability` 不作为长期 Scene Graph 字段保存；它们在当前规划或过滤阶段实时计算。
- 不使用实际无法可靠获得的物体真实质量、摩擦系数、精确质心或人工标注“最优姿态”作为核心网络输入。

#### 尚未最终确定的边界

- 当前主要讨论聚焦静态或非持续运动目标；是否正式扩展到动态抓取尚未确定。
- “放置”已出现在长程 Agent 示例和 Skill 框图中，但是否作为论文核心任务和正式评测阶段尚未确定。
- 是否在第二阶段一定使用 PPO，还是监督排序模型已经足够，需要通过实验决定。
- 是否增加 DQ-Net 式连续 High-Level Policy 尚未最终决定；当前建议是第一版不增加。

## 三、基础项目与当前代码基线

### 3.1 Agent 与 Skill 框架

当前项目的数据流为：

```text
自然语言
  -> vector-cli 或 vector-os-mcp
  -> Agent / VGG 任务分解与 tool use
  -> SkillRegistry 匹配和调用 Skill
  -> skill.execute(params, SkillContext)
  -> Service / Hardware Protocol
  -> 机器人或仿真器
```

关键代码事实：

- `vector_os_nano/core/skill.py`
  - 提供 `@skill` 装饰器、`Skill` 协议、`SkillRegistry` 和 `SkillContext`；
  - `SkillContext` 已支持 `arms`、`grippers`、`bases`、`perception_sources`、`services`、`world_model`、`calibration` 和 `config`；
  - 适合注册 Go2+ARX、感知服务、联合规划器和记忆服务。
- `vector_os_nano/hardware/base.py`
  - `BaseProtocol` 提供 `walk`、`set_velocity`、`get_position`、`get_heading`、`get_odometry` 和 `stop` 等接口；
  - Agent 和 Skill 不依赖具体底盘实现。
- `vector_os_nano/hardware/arm.py`
  - `ArmProtocol` 提供 `move_joints`、`move_cartesian`、`fk`、`ik` 和状态读取；
  - 可用于实现 ARX adapter，但当前仓库未发现 ARX 具体实现。

### 3.2 当前导航基线

- `vector_os_nano/skills/navigate.py`
  - 支持房间名称、坐标和导航栈路径；
  - 优先使用 `context.base.navigate_to`，在不同模式下调用导航栈或已有后备路径；
  - 可读取 Scene Graph / spatial memory 中的房间和位置数据。
- `vector_os_nano/hardware/sim/go2_ros2_proxy.py`
  - 发布 `/cmd_vel_nav` 和 `/goal_point`；
  - 订阅 `/state_estimation`；
  - 提供 `navigate_to` 与 Go2 导航栈之间的代理接口。
- `scripts/go2_vnav_bridge.py`
  - 桥接 Go2 仿真、ROS2 导航话题与状态估计；
  - 包含 `/state_estimation`、导航速度、目标点和 Scene Graph 可视化/发布相关逻辑。
- 当前对话将现有导航能力统称为 SysNav / FAR / NavStack 相关能力；最终实际采用的导航后端和版本仍需确认。

### 3.3 当前 Scene Graph 与 World Model

- `vector_os_nano/core/scene_graph.py`
  - 当前为三层结构：`RoomNode -> ViewpointNode -> ObjectNode`；
  - `ViewpointNode` 已包含位置、朝向、时间、场景摘要、可见物体和可选缓存帧；
  - `ObjectNode` 已包含类别、描述、置信度、房间、三维位置、属性和关联视点；
  - Scene Graph 支持持久化路径和空间记忆兼容接口。
- `vector_os_nano/core/world_model.py`
  - 保存动态 `ObjectState`、`RobotState` 和对象属性；
  - 可承载当前时刻的 bbox、点云引用和临时抓取数据，但长期结构化记忆由创新点二扩展。

### 3.4 当前抓取基线及缺陷

- `vector_os_nano/skills/pick_top_down.py`
  - 当前为顶部抓取；
  - 依赖 World Model 中已有物体位姿；
  - 使用 IK 和关节轨迹完成预抓取、下降、闭合和抬起；
  - 不等同于通用 6-DoF 抓取规划。
- `vector_os_nano/skills/mobile_pick.py`
  - 当前逻辑是先计算接近点并调用 `navigate_to`，再委托 `PickTopDownSkill`；
  - 尚未联合优化 base pose 与 grasp pose。
- `vector_os_nano/vcli/tools/sim_tool.py` 和 `scripts/go2_vnav_bridge.py`
  - 当前 Go2 带臂仿真主要接入 Piper 机械臂和 `/piper/*` ROS2 话题；
  - 实验室 Go2+ARX 需要替换或新增 adapter、模型和控制桥。

## 四、整体技术方案总览

### 4.1 总体分层

```text
自然语言用户指令
        |
        v
Agentic Task Orchestration Layer
任务理解 / 长程拆解 / Skill选择 / 执行监控 / 重规划
        |                    ^
        |                    | 状态、结果、错误原因
        v                    |
Shared World Model & Memory
Robot State / Task State / Task-Triggered Scene Graph
        ^                    ^
        | read/write         | read/write
        v                    v
Skill Layer
Perception / Navigation / Last-Mile View / Pair Planning /
Grasp / Verify / Transport / Place / Recovery
        |
        v
Reusable Service Layer
YOLO / SAM / RGB-D / SysNav / NavStack / MLLM /
GraspNet / AnyGrasp / GORM-like / Attention / IK / Collision
        |
        v
Robot Abstraction Layer
BaseProtocol / ArmProtocol / Gripper / Sensors / TF / ROS2
        |
        v
Go2 + ARX Robot Layer
Go2 12-DoF Whole-Body Policy / ARX IK / Gripper
        |
        +--------------------> feedback to Agent and Memory
```

### 4.2 统一执行逻辑

```text
Plan -> Execute Skill -> Observe -> Verify
                  |                    |
                  | failure            | success
                  v                    v
          Recover / Replan       Update Memory
                  |                    |
                  +-------> Continue Long-Horizon Task
```

### 4.3 三项创新的关系

1. **创新点一**解决“最终该站在哪里、采用哪个抓取位姿”的联合决策问题；
2. **创新点二**解决“粗导航结束后从哪里看目标，以及如何记住历史视点和 Pair”的记忆与最后一公里问题；
3. **创新点三**将前两项创新包装为 Agent 可调用能力，并通过共享状态、验证和恢复完成长程任务。

## 五、技术栈与依赖环境

### 5.1 已确认硬件与控制基础

| 类别 | 已确认内容 | 状态 |
|---|---|---|
| 四足平台 | Unitree Go2 | 已确认 |
| 机械臂 | ARX 系列机械臂 | 已确认；具体型号待确认 |
| 腿部控制 | Go2 12 个腿部自由度由已有底层策略控制 | 已确认 |
| 机械臂控制 | ARX 通过 IK 控制 | 已确认 |
| 协同能力 | 已有策略能够进行全身协同位姿跟踪 | 已确认 |
| 夹爪 | 项目需要夹爪开合与状态接口 | 功能需求已确认；具体硬件待确认 |
| 感知 | 方案需要 RGB-D/点云、里程计和导航环境感知 | 功能需求已确认；具体设备待确认 |

### 5.2 基础项目声明的软件栈

| 类别 | 当前仓库信息 |
|---|---|
| 语言 | Python 3.10+ |
| Agent | Vector OS Nano Agent / VGG / tool use |
| CLI / MCP | `vector-cli`、`vector-os-mcp` |
| 仿真 | MuJoCo；README 标注 MuJoCo 3.x |
| ROS | README 标注 ROS2 Jazzy；实际实验机 ROS2 版本待确认 |
| 感知可选依赖 | PyTorch、Transformers、OpenCV、Pillow、Open3D、RealSense 等 |
| IK / 运动学 | Pinocchio (`pin`)；现有 ArmProtocol 支持 FK/IK |
| Go2 相关 | CasADi、Pinocchio、Go2 ROS2 proxy 和导航桥 |
| 数据与配置 | NumPy、PyYAML、dotenv |
| 测试 | pytest |

### 5.3 计划接入或候选复用能力

| 功能 | 已讨论候选 | 当前状态 |
|---|---|---|
| 目标检测 | YOLO / YOLO-World | 候选，未最终选型 |
| 分割 | SAM / SAM3 | 候选，未最终选型 |
| 抓取候选 | GraspNet、AnyGrasp、GraspGen | 候选，未最终选型 |
| base 候选 | GORM-like 方法 | 方法方向已确认，具体实现待定 |
| 最佳视点选择 | Agent/MLLM 从最近关键帧中选取 | 方向已确认，模型待定 |
| 导航 | 当前 SysNav / FAR / NavStack 能力 | 复用方向已确认，具体部署配置待定 |
| Scene Graph | 基于当前 Room/Viewpoint/Object SceneGraph 扩展 | 已确认 |
| ARX 接口 | 参考 `arx5-sdk` 或实验室现有接口实现 ArmProtocol | 参考方向已讨论，最终方案待定 |

### 5.4 已讨论的关键参考工作

| 工作 | 与本项目的关系 | 开源状态（按对话检索时信息） |
|---|---|---|
| QuadWBG: Generalizable Quadrupedal Whole-Body Grasping | GORM 和 grasp-aware 高层规划的重要参考 | 未找到官方完整代码仓库 |
| DQ-Net / Whole-Body Coordination for Dynamic Object Grasping | GFM 的 Query-Key-Value 抓取融合与高层 PPO 训练参考 | 论文已确认；未找到官方代码 |
| GAMMA | graspability observation 和在线抓取姿态融合参考 | `github.com/user432/gamma` |
| UniLM-Nav | Last-K 视图记忆、MLLM 视图选择、最后一公里 base pose 推理参考 | 用户已提供论文 PDF |
| DovSG | 动态开放词汇 Scene Graph、记忆、导航和操作参考 | `github.com/BJHYZJ/DovSG` |
| RoboEXP | Action-conditioned Scene Graph 和交互探索参考 | `github.com/Jianghanxiao/RoboEXP` |
| HOV-SG | 层级开放词汇 3D Scene Graph 与语言导航参考 | `github.com/hovsg/HOV-SG` |
| ConceptGraphs | 开放词汇 3D Scene Graph 底座参考 | `github.com/concept-graphs/concept-graphs` |
| OpenFunGraph | 功能关系 Scene Graph 参考 | `github.com/ZhangCYG/OpenFunGraph` |
| VBC / Visual Whole-Body Control | 视觉全身控制和高低层控制参考 | `github.com/Ericonaldo/visual_wholebody` |
| UMI on Legs | 已确认的 Go2 Edu Plus + ARX5 开源硬件与全身控制参考 | `github.com/real-stanford/umi-on-legs` |
| ARX5 SDK | ARX5 C++/Python 控制接口、关节和笛卡尔控制参考 | `github.com/real-stanford/arx5-sdk` |

> 上表中的“参考”不表示已经决定直接采用其代码。除当前 Vector OS Nano 外，所有外部组件都需要进一步确认许可证、接口兼容性、硬件匹配和可复现性。

## 六、核心创新点一：Pairwise Base-Grasp Attention Planner

### 6.1 创新点名称

**Pairwise Base-Grasp Attention Planner**，也称 **Whole-Body Navigation-Grasp Joint Planner**。

### 6.2 解决的核心痛点

传统移动抓取通常先独立选择导航位置，再独立选择抓取位姿。这样会产生以下问题：

- 最佳导航站位不一定适合机械臂抓取；
- GraspNet/AnyGrasp 评分最高的 grasp pose 不一定能从当前站位到达；
- 两个分别最优的结果组合后可能发生 IK 失败、碰撞、路径代价过高或全身策略执行不稳定；
- 简单导航后抓取无法体现狗臂机器人的全身协同能力。

本创新点不分别选择 base 和 grasp，而是将二者绑定为完整 Pair，并结合目标、机器人、场景和任务上下文统一排序。

### 6.3 候选生成与 Pair 构造

#### 6.3.1 grasp candidates

- 使用 GraspNet、AnyGrasp 或 GraspGen 等现有模型从目标点云生成多个 6-DoF 抓取候选；
- 保留抓取网络给出的候选姿态及其原始分数；
- 最终采用哪一个抓取候选网络尚未决定。

#### 6.3.2 base candidates

- 使用类似 QuadWBG GORM 的方法建立 Go2+ARX 专用可达工作空间；
- 可通过 URDF、ARX IK、关节限制、碰撞检查和已有全身策略仿真获得可达位姿先验；
- 在线针对每个 grasp pose，依据 base-to-end-effector 可达关系反推出候选 base pose；
- 再结合当前地图和导航可达性进行过滤；
- 该方式优先于完全独立地在物体周围随机采样 base，再与所有 grasp 做巨大笛卡尔积。

#### 6.3.3 Pair Token

每个 Pair 表示一套完整方案，执行前可使用的特征包括：

- base pose（相对目标或当前机器人表示）；
- grasp pose（相对目标物体或 base 表示）；
- GraspNet/AnyGrasp 原始分数；
- GORM-like 可达先验分数；
- 导航路径长度或代价；
- 障碍物间距；
- IK 是否有解及关节裕量；
- 当前可获得的地图、对象和历史信息。

实际 rollout 后才能知道的抓取成功、滑落、真实跟踪误差等只能作为训练标签，不能作为在线输入泄漏给网络。

### 6.4 Hard Feasibility Filter

在 Attention 前使用确定性模块过滤明显无效 Pair：

- 导航不可达；
- base 位于障碍物或没有足够空间；
- ARX IK 无解；
- 超出关节限制；
- 已知自碰撞或环境碰撞。

这些约束由导航规划器、IK 和碰撞检测直接处理，不要求 Attention 重新学习确定性的物理规则。过滤结果通过有效 Mask 传入后续网络。

### 6.5 Attention 架构

#### 6.5.1 Context Tokens

上下文按来源编码：

- **Object Encoder**：物体 pose、bbox、点云或可获得的目标特征；
- **Robot Encoder**：当前 base pose、机械臂关节状态、本体状态；
- **Scene Encoder**：局部地图、障碍物、Scene Graph 目标邻域；
- **Task Encoder**：抓取、放置或其他任务语义。

#### 6.5.2 Context-Pair Cross Attention

- 每个 Pair Token 作为 Query；
- 多个 Context Tokens 作为 Key 和 Value；
- 每个 Pair 读取当前物体、机器人、场景和任务信息，生成上下文化表示 `H1...HN`；
- 这种设计适用于需要为 N 个 Pair 分别输出结果的排序任务。

#### 6.5.3 Pair Self-Attention

- 让不同候选 Pair 在同一场景中相互比较；
- 用于建模“抓取质量、导航代价、IK 裕量和场景适配”之间的相对权衡；
- 输出仍是一组逐 Pair 的上下文化特征。

#### 6.5.4 Actor MLP 与候选选择

- 每个 Pair 经过共享的 Actor MLP 得到 `S1...SN`；
- 无效候选通过 Mask 排除；
- Softmax 将有效候选分数转为当前候选集合中的选择概率；
- 训练阶段可按概率采样；
- 测试和实机部署时选择最高分 Pair，并再次执行精确 IK、碰撞和导航验证。

#### 6.5.5 不进行姿态平均

最终输出不是多个 base/grasp 姿态的加权平均，而是一个完整的 `(base*, grasp*)`。原因是候选可能分布在物体两侧或不同抓取模态中，直接平均可能生成位于障碍物内或旋转无效的姿态。

### 6.6 完整框图

```text
目标物体信息 -> Object Encoder ─┐
机器人状态 ---> Robot Encoder ──┤
地图/Scene Graph -> Scene Encoder├-> Context Tokens (Key / Value)
任务类型 ------> Task Encoder ───┘

GraspNet/AnyGrasp -> Grasp Candidates ─┐
GORM-like Module  -> Base Candidates  ─┤
导航/IK/碰撞检测 ----------------------┴-> Pair Construction
                                                |
                                      Hard Feasibility Filter
                                                |
                                      Pair Encoder: P1...PN
                                                | Query
                                                v
                              Context-Pair Cross Attention
                                                |
                                           H1...HN
                                                |
                                  Pair Self-Attention
                                                |
                                         Actor MLP
                                                |
                                    Scores S1...SN
                                       /             \
                           Stage-I Supervision     Stage-II PPO
                              labels -> loss       mask + softmax
                                       \             /
                                      selected Pair
                                   (base*, grasp*)
                                                |
                          NavStack + Whole-Body Policy + ARX IK
                                                |
                                 reward / outcome / feedback
```

### 6.7 训练阶段零：离线数据采集

这一步是制作监督数据集，不是网络参数训练。

对每个仿真场景：

1. 使用和部署阶段相同的感知、候选生成、Pair 构造和硬过滤流程；
2. 对所有 Pair 计算便宜的导航、IK、关节限制和碰撞结果；
3. 对部分有效 Pair 做局部抓取 rollout；
4. 对更少的一部分 Pair 做完整导航—到达—抓取 rollout；
5. 每次评估不同 Pair 时恢复相同机器人和物体初始状态，或使用并行仿真环境；
6. 保存场景上下文、Pair 特征、有效 Mask、是否测试 Mask 和执行结果。

不要求执行所有 Pair，更不要求在实机上穷举。优先执行以下组合：

- 当前评分较高的候选；
- 随机候选；
- 接近 IK、碰撞或可达边界的困难候选；
- 看起来较好但历史上失败的候选。

自动标签可包括：

- 最终是否成功抓住并抬起；
- 是否到达目标 base pose；
- 是否到达预抓取位姿；
- 是否接触物体；
- 是否闭合后滑落；
- 是否发生碰撞、摔倒或超时；
- 完成路径与时间。

建议的执行阶段标签为：

```text
阶段0：导航、IK或碰撞检查失败
阶段1：到达base，但末端未到预抓取位姿
阶段2：到达预抓取位姿，但未正确接触
阶段3：接触或闭合，但物体滑落
阶段4：成功抓住并稳定抬起
```

没有执行过的候选标记为未知，通过 `tested_mask` 排除出结果监督，不能伪造为成功或失败。

### 6.8 第一阶段：监督预训练

#### 训练范围

从 Object/Robot/Scene/Task Encoder 和 Pair Encoder 开始，经过 Cross Attention、Pair Self-Attention 和 Actor MLP，到 Pair 分数结束。

#### 冻结模块

- GraspNet/AnyGrasp；
- GORM-like candidate generator；
- 导航器；
- IK 与碰撞检查；
- 已有全身底层策略；
- 仿真环境本身。

#### 监督信号

- **成功二分类**：执行过的 Pair 是否成功；
- **相对排序**：同场景中成功 Pair 高于失败 Pair，执行阶段更深的失败 Pair可高于更早失败的 Pair；
- **阶段预测**：可作为辅助监督，不是最小版本必需；
- 两个成功 Pair 如果效果接近，不强制随意指定唯一最优；若实测时间、路径或结果存在明确差异，可用于排序。

最小可行训练版本为：

```text
Hard Mask + 成功二分类 + 同场景Pair相对排序
```

监督误差从 Actor 输出反向传播到 Actor MLP、Self-Attention、Cross-Attention 和各 Encoder；Attention 内部权重不需要单独人工标签。

#### 第一阶段作用

- 让 Actor 在 PPO 前具备基本可行性和候选排序能力；
- 缓解完整抓取成功奖励稀疏；
- 减少 PPO 初期反复选择明显失败 Pair；
- 提供可单独评估和消融的监督基线；
- 验证当前输入特征是否足以区分成功和失败方案。

### 6.9 第二阶段：PPO 微调

#### Actor

- 加载第一阶段训练好的全部 Actor 参数；
- 对当前候选集合输出概率；
- 训练时按照概率采样 Pair，部署时使用最高分 Pair；
- Pair 索引是高层离散动作。

#### 执行

- 选中 Pair 后，由 NavStack、已有全身策略、ARX IK 和夹爪完成宏动作；
- PPO 不需要通过 NavStack、IK 或物理仿真器直接反向传播；
- 它依据“选择该 Pair 的概率”和“执行后的奖励”更新 Actor。

#### Critic

- 输入建议为 Context Tokens 与 `H1...HN` 的池化结果；
- 预测当前场景预期能够获得的回报；
- 只辅助 PPO 训练，不负责选择 Pair；
- Critic 的具体结构仍待确认。

#### 奖励来源

- 成功抓住并抬起物体：主要正奖励；
- 到达 base pose、末端接近 pre-grasp：阶段性正反馈；
- 碰撞、摔倒、导航失败、超时：负反馈；
- 路径过长和耗时：较小代价。

具体奖励权重尚未确定，必须以仿真中可直接获得的状态为依据。

#### 更新与冻结

PPO 更新：

- Context/Pair Encoder；
- Cross Attention；
- Pair Self-Attention；
- Actor MLP；
- Critic MLP。

继续冻结：

- GraspNet/AnyGrasp；
- GORM-like 模块；
- 导航、IK 和碰撞检测；
- 已有全身策略；
- Agent LLM。

### 6.10 与 DQ-Net 的关系及区别

DQ-Net 的 GFM 使用当前物体 pose 与点云特征作为 Query，抓取 memory candidates 作为 Key/Value，生成融合抓取特征；该特征进入连续 High-Level Policy，高层策略通过 RL 反复输出末端增量和 base 速度。

本项目不同之处：

- 候选是 `(base, grasp)` Pair，而不是只有 grasp；
- 核心输出是完整 Pair 的排序和选择，不是融合后的原始姿态平均；
- 当前目标以事件级/规划级 Pair 选择为主；
- 已有导航器和全身策略负责闭环执行；
- 当前第一版不新增 DQ-Net 式连续高层策略。

如果未来增加连续 High-Level Policy，它将根据更新后的视觉、物体状态、机器人状态、末端误差等反复输出 base 速度和末端增量，但这会显著增加训练复杂度，当前不属于已确认的第一版方案。

### 6.11 创新价值与预期优势

- 从“导航后再抓取”提升为 base-grasp 一体化决策；
- 显式利用狗臂工作空间和全身执行结果；
- 将确定性几何过滤与学习型上下文排序结合；
- 输出物理可验证的候选锚点，便于解释、验证和失败回退；
- 可与 Agent、Scene Graph 和现有底层策略模块化集成。

### 6.12 依赖与约束

- 需要稳定的目标点云和坐标变换；
- 需要 Go2+ARX URDF、IK、关节限制和碰撞模型；
- 需要可调用的导航代价或可达性检查；
- 需要能够执行抓取物理的仿真环境，才能得到真正的抓取成功监督；
- 如果仿真不支持接触、夹持和物体抬起，只能训练“导航/IK/碰撞/跟踪可行性”，不能声称学到了真实抓取成功；
- PPO rollout 中必须保存当时的 Pair 特征、顺序和 Mask，更新时不能重新生成不同顺序的候选。

## 七、核心创新点二：Task-Triggered View-and-Grasp Memory Scene Graph

### 7.1 创新点名称

**Task-Triggered View-and-Grasp Memory Scene Graph**，中文暂称 **任务触发式视点-抓取记忆场景图**。

### 7.2 解决的核心痛点

- 粗粒度对象导航只能将机器人带到目标附近，最终视角可能遮挡、模糊或点云不完整；
- 导航终点不等于 manipulation-ready base pose；
- 首次抓取之后，Agent 无法复用同一物体或同类物体的历史最佳视点和成功 Pair；
- 如果为所有场景物体永久保存点云和候选抓取，Scene Graph 会过大且包含大量无用或过期数据。

### 7.3 核心设计原则

1. 保留原有 Room/Viewpoint/Object 轻量 Scene Graph；
2. 只有某个 ObjectNode 成为实际抓取目标时，才按需附加 Manipulation Memory；
3. 只保存目标相关 pose、bbox、裁剪点云引用、少量视点、少量 Pair 和执行历史；
4. 不保存长期 `reachability` 和 `stability` 字段；当前状态下重新计算；
5. 记忆用于提供视点与 Pair 候选种子，不能绕过当前感知、导航、IK 和碰撞验证直接执行。

### 7.4 目标节点的计划内容

#### 用户已明确希望保存

- 目标物体 pose；
- bbox；
- 目标裁剪 pointcloud；
- 少量 grasp candidates；
- 少量 base candidates；
- 实际执行成功和失败的 Pair history；
- 粗导航结束前的最近 5 帧画面及其视点信息。

#### 当前建议的数据组织方式

由于 base 与 grasp 的耦合是创新点一的核心，建议不再把二者作为互不关联的长期列表，而是按 Pair 保存：

```text
Target ObjectNode
├─ Current Geometry
│  ├─ pose
│  ├─ bbox
│  └─ cropped_pointcloud_ref
├─ Observation Memory
│  └─ RGB-D ref / base pose / camera pose / timestamp
├─ Pair Memory
│  └─ object-relative base / object-relative grasp /
│     planner score / executed flag / timestamp
└─ Attempt History
   └─ executed pair / success-failure / failure stage / timestamp
```

图片和点云属于大数据，当前建议由外部 Blob Storage 保存，Scene Graph 中仅保存引用、位姿、时间戳和结构化结果。该存储实现尚未最终确认。

### 7.5 最后五帧与最佳视点选择

参考 UniLM-Nav：

- 粗对象导航结束前维护 `K=5` 的短期观测缓冲；
- 每项观测包含 RGB-D、机器人 base pose、camera pose 和时间戳；
- Agent 调用 MLLM，从有编号的候选图像中选择一个最佳视图 ID；
- 选择依据主要为目标可见性和接近路径是否清晰；
- 视点选择与后续精确抓取规划分开调用，不合并为一次复杂 MLLM 输出。

当前建议不是机械保存五张几乎相同的连续视频帧，而是在位置或朝向发生变化时保存关键帧；具体关键帧阈值待确认。

### 7.6 `b_view` 生成

用户最初设想由 Agent/大模型选出最佳视角并得到相应基座位置。结合 UniLM-Nav 的结果，当前建议采用更易落地的方式：

1. MLLM 只返回 `selected_view_id`；
2. 读取该历史帧对应的 base pose，作为观察位置种子；
3. 在当前 costmap 中重新检查并吸附到安全可导航位置；
4. 使用几何方法设置朝向，使机器人面向目标物体；
5. 得到最佳观察位姿 `b_view`。

MLLM 是否进一步输出米制位置尚未最终确认；当前不建议让其自由输出未经验证的 `(x, y, yaw)`。

### 7.7 从最佳观察到最终抓取

```text
Agent查询目标ObjectNode
        |
        v
粗对象导航 / SysNav
        |
        v
维护Last-K RGB-D关键帧和机器人状态
        |
        v
MLLM选择selected_view_id
        |
        v
View-Pose Resolver生成b_view
        |
        v
NavStack最后一公里导航
        |
        v
到达b_view后重新获取RGB-D和点云
        |
        v
刷新目标pose / bbox / pointcloud
        |
        +------------------+
        |                  |
        v                  v
Fresh Grasp/Base       Historical Pair Seeds
Candidates             from Scene Graph
        |                  |
        +--------+---------+
                 v
       导航 / IK / 碰撞重新过滤
                 |
                 v
Pairwise Base-Grasp Attention Planner
                 |
                 v
        输出(b_grasp, grasp*)
                 |
                 v
全身策略 + ARX IK + Gripper执行
                 |
                 v
Verify成功/失败并更新Scene Graph
```

必须明确：

- `b_view` 是观察位置；
- `b_grasp` 是实际抓取站位；
- 到达 `b_view` 后必须重新感知，再调用创新点一；
- 历史 Pair 只能作为候选，必须变换到当前对象坐标并重新过滤。

### 7.8 第一次任务与重复任务

#### 第一次抓取

1. Scene Graph 中没有抓取历史；
2. 粗导航期间采集当前 Last-K 关键帧；
3. 选择 `b_view` 并刷新目标点云；
4. 运行新候选生成与 Pair Attention Planner；
5. 执行后写入最佳视点、Top-M Pair 和 Attempt History。

#### 再次抓取同一物体或同类物体

1. Agent 查询历史视点和 Pair；
2. 对同一物体，可把历史成功视点和 Pair 作为优先候选；
3. 对同类型新物体，历史 base/grasp 应保存为物体相对变换，再应用到新物体 pose；
4. 结合当前地图、点云和新生成候选重新过滤、排序；
5. 失败 Pair 用于避免无意义重复，成功 Pair 用于加速候选生成和规划。

同类型物体的匹配规则（类别、bbox 尺寸、形状特征或其他方式）尚未确定。

### 7.9 内存控制

已确定目标是稀疏、按需存储。当前建议包括：

- 每个目标只保存最新或代表性的裁剪点云；
- Last-K 使用固定长度环形缓冲；
- 长期只保留最佳视图和少量备选；
- Pair Memory 只保留 Top-M 和实际执行 Pair；
- Attempt History 使用固定长度队列；
- 所有记录带时间戳，过期数据需要刷新；
- 具体 K、M、历史长度和淘汰规则待确认。

### 7.10 创新价值与优势

- 将 Scene Graph 从纯导航语义记忆扩展为目标中心、任务触发的抓取记忆；
- 用历史视点弥合粗导航与高质量抓取感知之间的最后一公里；
- 用历史成功和失败 Pair 支持 Agent 长程任务中的经验复用；
- 通过稀疏存储避免对所有物体保存稠密抓取数据；
- 为创新点一提供更好点云与历史候选，为创新点三提供长期共享记忆。

### 7.11 依赖与约束

- 需要 Last-K RGB-D 与对应 base/camera pose 时间同步；
- 需要可靠 TF、相机标定和地图坐标；
- 需要目标检测/分割和裁剪点云；
- 需要 MLLM 结构化返回视图 ID；
- 需要当前 costmap 对历史视点重新验证；
- 场景变化后旧视点和 Pair 可能失效，不能直接执行；
- 如果所有历史视图均不可用，需要重新观察或简单主动扫描，具体恢复策略待确认。

## 八、核心创新点三：面向狗臂机器人的 Agent 长程任务执行框架

### 8.1 创新点名称

**Memory-Augmented Skill-Orchestrated Agent Framework for Long-Horizon Quadruped-Arm Manipulation**，中文暂称 **面向狗臂机器人长程任务的记忆增强型 Skill 编排 Agent 框架**。

### 8.2 解决的核心痛点

- 仅有导航算法和抓取算法不能自动完成自然语言长程任务；
- 一次性生成动作序列无法处理目标丢失、导航失败、抓取滑落等执行变化；
- 感知、导航、抓取、底层控制和记忆缺少统一接口；
- 前两个创新点需要以 Agent 可调用、可验证、可恢复的方式组织起来。

### 8.3 核心技术原理

Agent 只在语义和任务层决策，通过 Skill Registry 发现和调用能力。Skill 内部调用可复用 Service，最终通过硬件抽象层向 Go2+ARX 和已有底层策略发送结构化目标。执行结果返回 Agent 和共享世界模型，形成：

```text
任务规划 -> Skill调用 -> 机器人执行 -> 状态验证
    ^                                  |
    |                                  v
失败恢复 / 动态重规划 <- 错误原因与世界状态更新
```

### 8.4 整体框架

```text
Natural-Language Instruction
              |
              v
Agentic Task Orchestration
Instruction Understanding
Long-Horizon Task Decomposition
Skill Selection & Parameter Filling
Execution Monitor
Failure Recovery & Replanning
              |                         ^
              | structured calls        | status/result/error
              v                         |
Shared World Model & Scene Graph <-------+
Robot State / Task State / View & Pair Memory
              ^                         ^
              | read/write              | read/write
              v                         v
Skill Layer
├─ Perception Skills
├─ Navigation Skills
│  └─ Last-Mile View Navigation [Innovation 2]
├─ Mobile Grasp Skills
│  └─ Pairwise Base-Grasp Planner [Innovation 1]
├─ Transport & Place Skills
└─ Verification & Recovery Skills
              |
              v
Reusable Services
YOLO/SAM/RGB-D | SysNav/NavStack | MLLM
GraspNet/AnyGrasp | GORM-like | Attention | IK/Collision
              |
              v
Robot Abstraction
BaseProtocol | ArmProtocol | Gripper | Sensors | TF | ROS2
              |
              v
Go2 + ARX
Whole-Body Policy | ARX IK | Gripper
              |
              +------ feedback ------> Agent & Memory
```

### 8.5 Skill 组织

当前讨论中形成的最小能力集合为：

```text
detect_object
navigate_near_object
select_best_view
navigate_to_view_pose
refresh_target
plan_base_grasp_pair
execute_grasp
verify_grasp
navigate_to_place
execute_place
verify_place
update_memory
```

具体代码名称尚未最终确定，但职责边界应保持清晰：

- Perception Skill 调用 YOLO/SAM/RGB-D 等服务；
- Navigation Skill 完成粗导航；
- Last-Mile View Skill 读取创新点二并导航到 `b_view`；
- Pair Planner Skill 执行创新点一；
- Grasp Skill 调用 Pair Planner、底层策略、ARX IK 和夹爪；
- Verify Skill 不依赖 Agent 猜测，明确判断导航、抓取或放置是否完成；
- Recovery Skill 根据失败原因选择下一个视点、下一个 Pair 或重新感知。

### 8.6 两项前序创新在总体系统中的位置

#### 创新点一

可以作为独立的 `plan_base_grasp_pair` Skill 或 Planning Service，同时由更高层 `MobileGraspSkill` 调用。独立暴露有利于调试、消融和论文表达；最终代码封装方式待确认。

#### 创新点二

用户最初提出将抓取 Scene Graph 封装在 Navigation Skill 中。当前架构建议为：

- Scene Graph 作为 Agent、导航、抓取和验证共享的 Memory Service；
- Last-Mile View Selection 属于 Navigation Skill；
- Pair history 同时被 Grasp Planner 使用；
- Verify Skill 将结果写回 Scene Graph。

Scene Graph 的最终所有权是“导航内部”还是“共享 Memory Service”尚需用户最终确认；从长程任务需求看，共享方式更符合当前总体框图。

### 8.7 长程任务示例

自然语言：“把厨房桌上的瓶子拿到客厅桌上。”

```text
1. locate_object("bottle")
2. navigate_near_object("bottle")
3. select_best_view("bottle")
4. navigate_to_view_pose()
5. refresh_target_observation()
6. plan_base_grasp_pair("bottle")
7. execute_whole_body_grasp()
8. verify_object_held()
9. navigate_to_receptacle("living_room_table")
10. execute_place()
11. verify_place()
12. update_scene_graph()
```

若抓取失败：

```text
VerifyGrasp失败
  -> 写入失败Pair和失败阶段
  -> target_occluded: 重新选视点
  -> no_feasible_pair: 重新采集点云
  -> object_slipped: 尝试下一个Pair
  -> object_moved: 重新检测和规划
  -> Agent更新后续Skill序列
```

### 8.8 Skill 输出与反馈

当前建议所有 Skill 使用统一结果结构：

```text
status
result
confidence
failure_reason
world_model_update
suggested_recovery
```

该统一结构、字段名和状态枚举尚未在代码中最终确认。

### 8.9 Agent 与控制层级

```text
Agent：决定做什么和下一步调用哪个Skill
Skill：完成一个有明确前后条件的机器人子任务
Planner：计算具体b_view、b_grasp和grasp pose
底层策略：稳定跟踪base/末端目标
ARX IK：将末端目标转换为机械臂关节目标
```

当前建议 Agent 不直接输出速度、关节角或 ROS topic。ABot-CLAW 图中的自动 Python 代码生成可作为参考，但真实机器人第一版更适合结构化 Tool Call 和白名单 Skill。该执行形式尚未最终确认。

### 8.10 创新价值与优势

- 将 Agent、记忆、最后一公里视点导航、联合 Pair 规划和全身执行统一起来；
- 通过共享 Scene Graph 支持跨 Skill 和跨任务的长期信息复用；
- 通过 Verify 与 Recovery 将单向调用流程变为闭环长程执行；
- 保留现有导航、抓取和低层控制模块的可复用性；
- 使前两个算法创新能够在自然语言长程任务中形成系统级效果。

### 8.11 依赖与约束

- 每个 Skill 需要明确输入、输出、超时、前置条件、后置条件和失败原因；
- Task State 不能只依赖 LLM 对话上下文，应在 World Model 中持久化关键进度；
- Agent 不应绕过 Skill 直接调用高频硬件控制；
- 感知、导航、抓取和验证需要统一坐标系；
- 长程任务是否包含正式 place 阶段和哪些恢复分支仍需确认。

## 九、已确认的整体实现路径

用户提出“先将单 Go2 模型替换为当前 Go2+ARX 模型，调试导航，再加入抓取”的路线。当前分析确认该路线合理，建议按以下顺序推进：

### 9.1 阶段一：复现并冻结导航基线

1. 在当前 Vector OS Nano 中复现 Go2 导航；
2. 明确 SysNav/FAR/NavStack 输入输出、坐标系和话题；
3. 记录导航成功率、路径和终点误差作为基线。

### 9.2 阶段二：Go2+ARX embodiment adapter

1. 替换或新增 Go2+ARX 机器人模型；
2. 实现 ARX `ArmProtocol` 和夹爪接口；
3. 对齐 `base_link -> arm_base`、相机、雷达和末端坐标系；
4. 将已有全身策略接入 base/末端命令接口；
5. 机械臂保持收拢姿态，先验证带臂后的导航稳定性。

### 9.3 阶段三：固定站位抓取

1. 接入 RGB-D 目标检测、分割和点云；
2. 接入一个抓取候选网络；
3. 验证 ARX IK、夹爪和单次固定 base 抓取；
4. 建立抓取成功验证。

### 9.4 阶段四：base-grasp 候选与非学习基线

1. 建立 Go2+ARX GORM-like 可达先验；
2. 从 grasp 反推 base candidates；
3. 实现 Pair 构造、导航/IK/碰撞硬过滤；
4. 先使用人工规则或简单评分完成端到端移动抓取；
5. 该基线用于后续 Attention 对比，不作为最终创新算法。

### 9.5 阶段五：创新点一训练与集成

1. 自动采集 Pair 数据和结果标签；
2. 监督预训练 Pair Attention Actor；
3. 与人工评分、独立 base/grasp 选择进行对比；
4. 在条件允许时进行 PPO 微调；
5. 将选中 Pair 交给导航器和已有全身策略执行。

### 9.6 阶段六：创新点二

1. 扩展目标 ObjectNode 的按需 Manipulation Memory；
2. 实现 Last-K RGB-D 关键帧缓冲；
3. 实现 MLLM 最佳视图 ID 选择；
4. 实现 `b_view` 解析、最后一公里导航和到达后重新感知；
5. 保存 Top-M Pair 与成功/失败历史；
6. 在重复任务中把历史 Pair 作为新候选种子。

### 9.7 阶段七：创新点三

1. 将感知、导航、视点、Pair 规划、抓取和验证封装为 Skill；
2. 建立 Task State、Skill Result 和失败原因；
3. 接入 Agent 的长程任务拆解和 Skill 调用；
4. 完成失败恢复和 Scene Graph 更新闭环；
5. 进行仿真和实机长程任务评测。

## 十、预期成果与实验框架

### 10.1 预期成果

- 一个可通过自然语言控制 Go2+ARX 进行导航和抓取的 Agent 系统；
- 一套可复用的 Go2+ARX Vector OS Nano adapter 和 Skill 接口；
- Pairwise Base-Grasp Attention Planner；
- 任务触发式视点-抓取记忆 Scene Graph；
- 具有验证、恢复和记忆更新的长程任务执行框架；
- 仿真与真实 Go2+ARX 实验；
- 面向 SCI 论文的算法、系统、消融和对比结果。

当前没有确认任何目标数值、成功率门槛、数据规模或投稿期刊。

### 10.2 创新点一建议消融

以下实验在对话中已经讨论，具体配置待确认：

- 导航与抓取独立选择；
- 人工加权 Pair 评分；
- 监督训练 Pair Attention；
- 监督训练 + PPO 微调；
- 无 Pair Self-Attention；
- 无场景/机器人/历史 Context；
- 固定身体抓取与已有全身策略执行对比。

建议指标来源于当前任务可测量结果：

- 完整抓取成功率；
- 到达后条件抓取成功率；
- 导航路径长度和时间；
- IK/碰撞失败率；
- 重规划次数；
- 未见物体和未见场景泛化。

### 10.3 创新点二建议消融

```text
最终导航帧直接进入抓取
+ Last-K最佳视点选择
+ 历史Pair记忆复用
完整Scene Graph + Pair Attention Planner
```

建议观察：

- 最后一个导航帧与 MLLM 选中帧的视图质量差异；
- 到达 `b_view` 后目标检测和点云质量；
- 最后一公里导航成功率；
- 抓取规划时间；
- 重复任务成功率；
- 历史 Pair 命中、失效和重新过滤情况。

### 10.4 创新点三建议消融

- 无长期 Scene Graph memory；
- 无 Pair Attention Planner；
- 一次性开环 Skill 序列；
- Plan-Execute-Verify-Replan 闭环；
- 无失败恢复；
- 完整 Agent 系统。

建议指标包括：

- 长程任务整体成功率；
- 各子任务条件成功率；
- 平均 Skill 调用次数；
- 平均重试和重规划次数；
- 失败恢复成功率；
- 系统规划与执行耗时。

## 十一、已明确的规则与约束

1. base 和 grasp 必须作为 Pair 联合评价，不能分别选各自最高分后直接组合。
2. 不将多个跨模态姿态直接加权平均为最终物理姿态。
3. 硬导航、IK 和碰撞约束优先由确定性模块处理。
4. Attention 使用执行前可获得的信息；执行后结果只能作为标签或奖励。
5. 未执行 Pair 必须标记未知，不能伪造监督标签。
6. Scene Graph 只对抓取目标按需增加重型操作记忆。
7. Scene Graph 中的历史 Pair 必须以物体相对形式迁移，并在当前场景重新验证。
8. `b_view` 与 `b_grasp` 是两个不同目标，不能混为一谈。
9. 到达最后一公里观察位置后必须重新感知目标，再进入抓取规划。
10. Agent 不直接替代导航器、IK 或高频全身控制器。
11. 现有全身策略在联合规划器训练中原则上冻结；是否后续联合微调尚未确认。
12. 所有大模型输出的视点、位置或动作在执行前必须转换为结构化结果并经过几何或安全验证。

## 十二、待确认事项清单

### 12.1 项目与论文定位

1. `Q-GRASP` 是否作为最终项目和论文方法名？
2. 暂定论文标题 `Q-GRASP: Grasp-aware Scene Graphs and Pair-wise Base-Grasp Attention for Agentic Quadruped-Arm Mobile Manipulation` 是否采用？
3. 论文核心任务是否仅为 pick，还是正式包含 pick-and-place 和运输？
4. 当前论文是否明确限定静态目标，还是包含动态物体？
5. 目标 SCI 期刊或会议、计划时间和实验规模尚未确定。

### 12.2 硬件与底层控制

6. 实验室 ARX 机械臂的准确型号是否为 ARX5？
7. 夹爪型号、控制接口和可读取状态尚未确认。
8. RGB-D 相机型号、安装位置（base/腕部）、帧率和标定方式尚未确认。
9. LiDAR、SLAM、里程计和导航计算平台的实际配置尚未确认。
10. 已有全身策略使用的仿真器、观测、动作、命令空间和控制频率尚未记录。
11. 全身策略最终接收 base velocity、SE(2) waypoint、EE pose、模式指令中的哪些命令尚未最终确认。
12. ARX IK 与全身策略之间的时序、坐标系和安全约束尚未确认。

### 12.3 当前项目迁移

13. 最终采用 SysNav、FAR、Nav2 中哪一套或怎样组合，需要以实际部署环境确认。
14. Go2+ARX 仿真模型、URDF/MJCF 和现有训练项目如何接入 Vector OS Nano 尚未确定。
15. 是否直接基于 `arx5-sdk` 实现 `ArmProtocol`，还是复用实验室已有 ROS/SDK 接口尚未决定。
16. 当前 Piper 相关仿真是否保留为基线，还是直接替换为 ARX 尚未决定。
17. ROS2 发行版和实验机系统环境尚未确认。

### 12.4 感知与抓取候选

18. 目标检测最终采用 YOLO、YOLO-World 或其他模型尚未决定。
19. 分割最终采用 SAM、SAM3 或其他模型尚未决定。
20. 抓取候选最终采用 GraspNet、AnyGrasp、GraspGen 或组合方式尚未决定。
21. 是否能够访问抓取网络中间 3D 特征尚未确认；核心方案不应依赖该特征。
22. 仿真环境是否支持真实接触、夹持、滑落和抬起标签尚未确认。
23. 抓取成功的仿真与实机验证标准尚未确定。

### 12.5 Pair Attention Planner

24. GORM-like 可达工作空间的具体构建方法、采样维度和数据规模尚未确定。
25. 每个场景保留的 grasp、base 和 Pair 数量上限尚未确定。
26. Pair Token 最终字段、坐标表示和归一化方式尚未确定。
27. Context Encoder 是否使用原始地图 patch、Scene Graph tokens 或两者组合尚未确定。
28. Cross Attention 和 Pair Self-Attention 的层数、维度、Head 数尚未确定。
29. 第一阶段是否加入阶段预测辅助头尚未决定；最小方案是成功二分类加相对排序。
30. 监督数据中局部 rollout 与完整导航抓取 rollout 的比例尚未确定。
31. 成功与失败样本不平衡的采样和损失权重尚未确定。
32. PPO 是否为最终必要模块，需要通过监督模型效果决定。
33. PPO reward 各项权重、宏动作周期、并行环境数和 rollout 设置尚未确定。
34. Critic 网络结构和共享哪些 Actor 特征尚未确定。
35. 是否未来增加连续 High-Level Policy 尚未确定；当前第一版建议不增加。

### 12.6 Scene Graph 与最后一公里

36. Manipulation Memory 是扩展 ObjectNode attributes、增加新节点类型，还是使用外部数据库关联，尚未确定。
37. 大图像和点云的 Blob Storage 格式、路径、压缩和生命周期尚未确定。
38. Last-K 是否固定为 5，以及关键帧触发条件尚未确定。
39. 长期保存所有 5 个视图还是只保留最佳视图和备选，尚未确定。
40. `b_view` 是否直接复用历史 base pose、进行 costmap 吸附，还是复现 UniLM-Nav 的几何 base-pose reasoning，尚未最终确认。
41. 最佳视点选择使用哪一个 MLLM/Agent 模型尚未确定。
42. 同类物体检索使用类别、bbox、点云特征或其他相似度规则尚未确定。
43. Pair Memory 的 Top-M、Attempt History 长度和淘汰策略尚未确定。
44. 当 Last-K 中没有清晰视图时，原地扫描、下一个视点或主动局部探索的恢复策略尚未确定。
45. Scene Graph 最终作为共享 Memory Service，还是部分封装在 Navigation Skill 内，需最终确认。

### 12.7 Agent 长程框架

46. Agent 使用的 LLM/VLM/MLLM 提供方和模型尚未最终选定。
47. Agent 最终输出结构化 Tool Call，还是受限 Python Skill 代码，尚未最终确认；当前建议优先结构化调用。
48. Skill 的最终名称、输入 schema、状态枚举和错误码尚未确定。
49. Execution Monitor 和 Task State 的持久化格式尚未确定。
50. 导航、抓取和放置的前置/后置条件与超时尚未确定。
51. VerifyGrasp、VerifyPlace 和 object-held 的实机判据尚未确定。
52. 第一版必须支持的失败恢复分支尚未最终收敛。

### 12.8 实验与交付

53. 仿真场景、训练对象、未见对象和未见场景划分尚未确定。
54. 实机测试对象类别、摆放方式、房间数量和任务列表尚未确定。
55. 与哪些公开方法进行完整定量对比尚未最终决定。
56. 是否开源 Go2+ARX adapter、数据集、训练代码和 Scene Graph 扩展尚未决定。
57. 预期成功率、实时性和内存占用等验收阈值尚未确定。

## 十三、版本变更规则

后续任何以下变化都必须更新本文档版本并写入变更记录：

- 三个核心创新点的名称、输入输出或核心机制改变；
- 硬件平台、机械臂型号、传感器或底层策略改变；
- 导航、抓取候选、MLLM 或 Agent 模型选型确定或替换；
- Scene Graph 数据结构和持久化策略确定；
- 训练阶段、标签、损失或 PPO 方案改变；
- 项目边界、论文任务和评测指标改变。

### 13.1 变更记录

| 版本 | 日期 | 变更内容 |
|---|---|---|
| V1.0 | 2026-07-15 | 基于当前对话全量历史和当前仓库代码建立首个项目基准版本 |

