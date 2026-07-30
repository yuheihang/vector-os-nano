# VectorEngine — 统一执行引擎

## 概述

VectorEngine 是 Vector OS Nano 的唯一执行引擎。v2.0 起，CLI 和 MCP 共用同一引擎：

```
vector-cli  ─┐
             ├→ VectorEngine → VGG / tool_use → skill.execute()
vector-os-mcp┘
```

用户说自然语言，AI agent 通过工具系统同时控制机器人、编辑代码、诊断问题 —— 一个 session 里完成所有事。

```
用户: "探索的时候狗在转角撞墙"
  ↓
AI Agent (VectorEngine)
  ├── file_read("go2_vnav_bridge.py")     → 读路径跟随代码
  ├── file_edit(old="0.6", new="0.4")     → 改转弯速度
  ├── skill_reload("walk")                → 热加载，不用重启
  ├── explore()                           → 重新跑探索
  └── 回复: "改了转弯速度，重新探索中"
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│  vector-cli (vcli/cli.py)                                       │
│                                                                 │
│  用户输入 ──→ IntentRouter (意图分类)                             │
│                  │                                              │
│                  ↓                                              │
│              VectorEngine.run_turn_native() / run_turn()          │
│                  │                                              │
│                  ├── DynamicSystemPrompt                         │
│                  │     ├── 角色设定 (缓存)                        │
│                  │     ├── 工具使用说明 (缓存)                     │
│                  │     ├── 硬件/技能/世界模型 (静态)               │
│                  │     └── [机器人状态] (每次刷新)                  │
│                  │           位置、房间、SceneGraph、              │
│                  │           导航状态、探索进度                     │
│                  │                                              │
│                  ├── CategorizedToolRegistry                     │
│                  │     ├── code:    文件读写编辑、bash、搜索        │
│                  │     ├── general: web_fetch                     │
│                  │     ├── robot:   场景图/世界查询 (+臂控技能)     │
│                  │     ├── diag:    ROS2话题/节点/日志、导航/地形   │
│                  │     ├── sim:     start/stop_simulation         │
│                  │     └── system:  状态、热加载、foxglove         │
│                  │                                              │
│                  ├── ToolHookRegistry                            │
│                  │     ├── pre_hook: 执行前回调                   │
│                  │     └── post_hook: 执行后回调(验证/统计)         │
│                  │                                              │
│                  ├── LLM 后端 (Anthropic / OpenRouter / 本地)     │
│                  ├── 权限系统 (7层检查)                            │
│                  └── Session (JSONL 持久化)                       │
└─────────────────────────────────────────────────────────────────┘
```

裸 `vector-cli` 的自然语言 REPL 默认使用 `run_turn_native()`；`run_turn()` 是兼容路径。
二者复用同一个 VectorEngine、world 注册表和验证命名空间。下述分类路由流程主要描述
通用/兼容 tool-use 路径；native producer 则直接组合当前 world 的 action surface 与合成的
`verify`、`finish` 工具。

## Tool Call 完整流程

```
1. 用户输入自然语言
2. IntentRouter 关键词分类 → 选择相关工具类别
   "去厨房" → robot+diag (省 68% token)
   "改代码" → code+system (省 72% token)
   "你好"   → 全部工具 (无法判断意图)
3. VectorEngine 序列化:
   - system_prompt (DynamicSystemPrompt 刷新机器人状态)
   - messages (对话历史)
   - tools (只发选中类别的工具 schema)
4. LLM 返回 tool_use 调用
5. 引擎分区执行:
   - 只读 + 并发安全 → 并行 (ThreadPoolExecutor, 10 workers)
   - 写入 / 电机控制 → 串行
6. 每个工具执行:
   a. pre_hook 触发 (日志/预检)
   b. 权限检查 → allow / deny / ask(用户确认)
   c. tool.execute(params, context) → ToolResult
   d. post_hook 触发 (验证/统计)
   e. 电机技能 → 自动附加执行后状态(位置/房间)
7. 结果追加到 session
8. 循环回步骤 3，直到 LLM 返回 end_turn
9. 最终文本渲染到 CLI 面板
```

## CategorizedToolRegistry — 分类工具注册表

### 设计思路

继承自 `ToolRegistry`（完全向后兼容）。核心能力：
- 工具按类别分组管理
- 运行时动态启用/禁用整个类别
- 配合 IntentRouter 按意图只发送相关工具

```python
class CategorizedToolRegistry(ToolRegistry):
    _categories: dict[str, list[str]]   # 类别 → [工具名列表]
    _disabled: set[str]                 # 已禁用的类别

    def register(self, tool, category="default") -> None
    def enable_category(self, category: str) -> None
    def disable_category(self, category: str) -> None
    def to_anthropic_schemas(categories=None) -> list[dict]  # 可按类别过滤
    def list_categories(self) -> dict[str, list[str]]
```

### 工具类别

| 类别 | 工具 | 用途 |
|------|------|------|
| `code` | file_read, file_write, file_edit, bash, glob, grep | 代码读写编辑 |
| `general` | web_fetch | 网页抓取 |
| `robot` | world_query, scene_graph_query（+ 臂控时 10 个技能工具） | 空间查询 + 机械臂控制 |
| `diag` | ros2_topics, ros2_nodes, ros2_log, nav_state, terrain_status | ROS2 诊断 |
| `sim` | start_simulation, stop_simulation | 仿真管理 |
| `system` | robot_status, skill_reload, open_foxglove | 系统状态与热加载 |

### 扩展策略

| 阶段 | 策略 | 效果 |
|------|------|------|
| v1（当前） | 全部启用，IntentRouter 按意图路由 | 平均省 52% token |
| v1.1 | 延迟 schema — 先发名字，LLM 需要时再请求完整定义 | 再省 60% |
| v2 | 外部插件 — pyproject.toml entry_points 注册第三方工具 | 无限扩展 |

### 添加新工具（开发者工作流）

```python
# 1. 新建文件: vcli/tools/my_tool.py
@tool(name="my_tool", description="...", read_only=True, permission="allow")
class MyTool:
    input_schema = { "type": "object", "properties": { ... } }
    def execute(self, params, context) -> ToolResult: ...

# 2. 在 vcli/tools/__init__.py 中:
#    - discover_all_tools() 里加 import + 实例化
#    - _TOOL_CATEGORIES["my_category"] 里加工具名

# 完成。不需要改引擎、后端、权限、或任何其他文件。
```

## IntentRouter — 意图路由器

零成本关键词匹配，在 LLM 调用前选择相关工具类别：

```python
class IntentRouter:
    def route(self, user_message: str) -> list[str] | None:
        # 返回类别列表，或 None（发全部工具）

# 规则示例:
# "改"/"edit"/"code"/"bug"  → ["code", "system"]
# "去"/"走"/"explore"       → ["robot", "diag"]
# "topic"/"log"/"为什么"    → ["diag", "system"]
# "你好" (无关键词匹配)     → None → 全部工具
```

Token 节省效果：

| 场景 | 改前 (19 内置工具全发) | 改后 (路由) | 节省 |
|------|----------------------|------------|------|
| "我在哪" | ~2500 tokens | ~800 tokens | 68% |
| "改速度" | ~2500 tokens | ~700 tokens | 72% |
| "你好" | ~2500 tokens | ~2500 tokens | 0% |
| 平均 | ~2500 tokens | ~1200 tokens | ~52% |

## Tool Protocol — 工具协议

每个工具实现这个接口（Protocol 类型，不需要继承）：

```python
class Tool(Protocol):
    name: str                           # 工具名
    description: str                    # LLM 看到的描述
    input_schema: dict[str, Any]        # JSON Schema 参数定义

    def execute(params, context) -> ToolResult          # 执行
    def check_permissions(params, context) -> PermissionResult  # 权限检查
    def is_read_only(params) -> bool                    # 只读？
    def is_concurrency_safe(params) -> bool             # 可并发？
```

`@tool` 装饰器自动注入 permissions、read_only、concurrency 的默认实现。

## SkillWrapperTool — 技能包装器

Robot skill（`@skill` 装饰器）自动包装为 LLM tool：

```
@skill(aliases=["stand", "站"]) class StandSkill  →  SkillWrapperTool("stand")
@skill(aliases=["navigate"])    class NavigateSkill →  SkillWrapperTool("navigate")
```

包装器增加的能力：
- **电机检测**: effects 中包含 "move"/"navigate"/"arm" → 需要用户授权
- **执行后状态**: 电机技能执行后，自动附加当前位置/房间到结果
- **恢复提示**: 失败时根据 diagnosis_code 给出下一步建议

```
成功: "Skill 'navigate' succeeded. Data: {room: kitchen}
       State: pos=(16.8, 2.3) room=kitchen"

失败: "Skill 'navigate' failed. (room_not_explored)
       Suggested: Room not explored yet. Run the explore skill first.
       Current state: {position: [10.0, 5.0], room: hallway}"
```

已知的恢复提示映射：

| diagnosis_code | 提示 |
|---------------|------|
| no_base | 没有连接机器人，用 start_simulation 启动仿真 |
| unknown_room | 房间不存在，用 scene_graph_query 查看可用房间 |
| room_not_explored | 房间未探索，先运行 explore |
| navigation_failed | 导航失败，用 nav_state 检查导航栈状态 |
| no_vlm | VLM 不可用，检查 Ollama 是否运行 |
| camera_failed | 摄像头未连接，用 robot_status 检查硬件 |

## Native CLI 导航契约（P1）

有移动底座时，native producer 只向模型暴露两个无歧义的导航 schema：

| native 工具 | 使用条件 | 独立验收 |
|---|---|---|
| `navigate_room(room: str)` | 用户给出命名房间，如 `dining room`、`厨房` | `in_room("canonical_room")` |
| `navigate_xy(x: float, y: float)` | 用户明确给出坐标，或上游感知/规划工具返回坐标 | `at_position(x, y)` |

`navigate` 仍是内部正式 `NavigateSkill` 的名称，不再作为模糊的 native 公共 schema。
两个 native 适配器都委托该正式技能执行；其中命名分支统一通过
`navigation/room_resolver.py::RoomResolver` 规范化中英文别名、检查可用房间并读取中心。
native 层不保存第二份坐标表，也不直接绕过技能调用 base。命名解析失败返回
`unknown_room` 和当前可用房间，保持停车，不得再调用 `navigate_xy` 猜一个坐标。

房间词表和执行解析来自同一个 live SceneGraph：

- `known_layout`：可见已装载到 SceneGraph 的全部先验房间节点。
- `unknown_exploration`：只可见在线发现的房间；不得读取未发现房间的布局中心。

native prompt 每轮列出这组 canonical room IDs，并明确要求：命名目的地调用
`navigate_room`，绝不发明房间坐标；只有显式坐标或上游工具给出的坐标才调用
`navigate_xy`。原始 `room_layout.yaml` 不直接交给模型。

坐标导航的 transport 到达半径与 `at_position` 验证统一为 0.5 m，不能再出现底层已
返回成功、验证层却永久失败的死区。停滞检测按一段时间窗口内的累计进展判断：接近
目标、机体实际平移或实际转动都算进展，允许大角度对齐和 localPlanner 为绕障先横移/
短暂远离目标。这里的低速可能来自转向、曲率、障碍或接近目标，是通用观测条件，不是
门区固定限速；连续三次相同导航/验证失败后 native runner 停止恢复并报告失败。交互
trace 会把连续同目标的重复尝试压缩为一行并标出 attempts，完整尝试细节仍保留在诊断
日志中。

在 `known_layout` 仿真中，schema-v2 配置中的房间 polygon 与门的中心、宽度、法向和
两侧 standoff 是确定性拓扑先验：启动时会覆盖旧持久化文件中漂移的先验几何值，但
保留已学习的物体、房间描述、访问历史以及门观测记录。在线 observed 门不会成为
已知户型的捷径。命名导航先生成可信门链，再逐段提交
`door_pre → door_center → door_post → … → room_goal`；任何一段无路、超时或落点超差
都会停车，且不得跳过该门直达目标。房间 `center` 保留语义含义；若中心被家具占用，
执行层使用同一房间 polygon 内的 `navigation_goal`。例如 dining room 的语义中心是
`(3.0, 7.5)`，实际安全落点是 `(4.8, 6.0)`。
到 `door_pre` 的房间内接近段允许控制器正常对齐；只有
`door_pre → door_center → door_post` 的实际门槛穿越禁止倒车，并执行严格门点/segment
ACK。所有门点均为 `speed_limit_mps=null`，不施加额外固定低速上限；localPlanner
使用与普通导航相同的正常自适应速度。障碍、曲率、转向以及接近每个 waypoint 时的
通用减速仍然有效。
`unknown_exploration` 使用独立持久化文件，不加载这些先验。

### P2 运行时导航与可视化契约

已知户型不是 7 门的“所有房间经 hallway”模型，而是与 MJCF 一致的 8 房间、9 门
拓扑。除 hallway 两侧门外，还包括 living room ↔ dining room
`(3.0, 5.0)`、kitchen ↔ study `(17.0, 5.0)` 两扇直连门；y=10 的三个开口分别是
master bedroom ↔ dining room、guest bedroom ↔ study、bathroom ↔ hallway。因而
living room ↔ dining room 必须使用直连门，fresh hallway → dining room 则只使用
hallway ↔ dining room 门。旧 7 门先验会制造用户在 RViz 中看到的错误远路。

一条门链以 `room_route_timeout=360 s` 作为最小总墙钟预算，并按拓扑复杂度和机体
折线路程有界扩展：`min(1200, max(360, door_count×300, polyline_m×55))`。执行器再按
当前位置到剩余 waypoint 的折线距离分配各段时间，同时为每个后续段保留
`room_route_min_segment_timeout=35 s`。因此三扇门、十个分段不会再让未来段的保底时间
耗尽当前段预算；绝对上限仍为 1200 s。真正无进展由 30 s stall watchdog 提前
fail closed 并停车，门点容差不会因预算扩展而放宽。

应用层坐标统一是 `map` 中的 Go2 机体中心：房间点、显式 XY、`get_position()` 与
deterministic verify 都遵守这个定义。CMU 栈的 `/state_estimation` 则发布
`child_frame_id=sensor`，传感器相对机体前置 0.30 m、上置 0.20 m；proxy 在读取时只做
一次 sensor → body 转换，并把机体目标投影成 FAR 所需的传感器目标。FAR 终止点同样用
body → sensor 转换。localPlanner `/path` 的 `map` 坐标直接使用，`sensor` 坐标以传感器
为原点，`vehicle/base_link` 以机体为原点；不认识的 frame 会被拒绝并清空，避免悄悄
错 0.30 m 或把局部路径当全局路径。

RViz 产品层只默认显示四条有明确来源的 Path：

| Topic | 含义/颜色 | 生命周期 |
|---|---|---|
| `/scene_graph/door_path` | 门级拓扑，紫 | `RELIABLE + TRANSIENT_LOCAL + KEEP_LAST(1)` |
| `/far/global_path` | FAR 全局路径，蓝 | 同上 |
| `/local_planner/path` | localPlanner 局部路径，绿 | 同上 |
| `/nav/executed_path` | 机体实际轨迹，黄 | 同上 |

四路 publisher 在启动时保留空 Path，新目标开始前先清旧状态，并在成功、失败、取消、
reset、nav gate 关闭或断开时再次保留空 Path；晚启动的 RViz 因而收到的是空终态，不会
复活上次路线。第三方原始 `/path`、`/viz_path_topic`、`/exploration_path` 和
`/global_path_full` 在默认 RViz 配置中关闭。

动态验收中，“非空 `/registered_scan` + 非空 `/local_planner/path` + 最终到达”证明
传感器障碍数据和 localPlanner 确实参与本次导航，不是只画了一条 SceneGraph 直线。
这仍不是受控移动障碍避让试验；要验证后者还需布置可复现障碍并记录安全间距与绕行
轨迹。

导航结果使用统一 envelope，便于 trace、日志和后续验证关联：

```json
{
  "goal_id": "G123",
  "goal_type": "room",
  "requested_room": "dining room",
  "canonical_room": "dining_room",
  "target_xy": [4.8, 6.0],
  "semantic_center_xy": [3.0, 7.5],
  "source": "scene_graph",
  "planner": "far_segmented",
  "arrived": true,
  "verification_mode": "polygon",
  "navigation_stats": {
    "nonzero_cmd_count": 417,
    "cmd_vel_count": 417,
    "nonzero_cmd_duration_s": 21.3,
    "moved_distance_m": 5.8,
    "distance_travelled_m": 5.8,
    "actual_velocity_observed": true,
    "actor_caused": true
  }
}
```

`in_room(room_id)` 与导航共用 `RoomResolver`。它按 `room_at`、polygon、bounds 的顺序
判断房间归属；几何信息不足时才降级到最近中心，并把
`verification_mode=nearest_center` 留在可观察结果中。未知房间、无底座或不可用几何都
fail closed。

actor causation 也以 `goal_id` 为边界：proxy/bridge 必须记录属于该目标的实际非零
path-following `cmd_vel`，并结合底座位移，才能把动作评为 CAUSED。仅发布目标、零速度、
后台自然漂移或其他目标的速度都不能归因给当前动作。真实移动但未通过 `in_room` /
`at_position` 只能是 RAN；动作证据和确定性 post-condition 都通过后才是 GROUNDED。
`navigation_stats.actor_caused` 同样要求实际命令和超过抖动阈值的位移；仅观察到非零
命令时只设置 `actual_velocity_observed=true`。

## 复杂 NL 任务的分解路径 — cognitive/ VGG 层

单工具调用（"pick the cup"）由 VectorEngine 的 agent 循环直接处理。**多步复杂任务**的自然语言分解由 `vcli/cognitive/` 负责：

```
vcli/cognitive/
├── vgg_harness.py        # VGG 主循环入口
├── goal_decomposer.py    # 将 NL 目标拆解为子步骤计划
├── goal_executor.py      # 逐步执行计划，调用 VectorEngine
├── goal_verifier.py      # 验证每步结果是否达标
├── strategy_selector.py  # 根据硬件/世界状态选择执行策略
├── capabilities/         # 能力映射（什么机器人能做什么）
└── worlds/               # 世界模型适配（sim / real / ros2）
```

该层实现 **decompose → plan → execute → verify** 循环，是"自然语言控制一切"北极星目标的核心执行路径。详见 [docs/ARCHITECTURE.md](ARCHITECTURE.md)。

## 非交互验收契约 — `-p / --json / VECTOR_VERDICT` (R2a acceptance instrument)

`cli.main` 既是 REPL，也是**机器可验收的验收面**。这是本项目 #1 历史失败（能力靠 `~/sandbox` 脚本"验证"、绕过产品；347 个测试只有 2 个碰 `cli.main`）的根治：引擎本就计算的诚实判定
`evidence_passed(trace, verify_oracle_names(agent, engine))` 现在能作为机器信号逃出 `cli.main`。

```
python -m vector_os_nano.vcli.cli -p "<prompt>" --json
```

- **`-p / --print TEXT`** — 跑 **一个** turn（不进 REPL）后退出。
- **`--json`** — 在 stdout 打印**恰好一行** `VECTOR_VERDICT {<json>}`（固定 sentinel）；所有 Rich/banner 改走 **stderr**。
- 该 turn 经 `engine.vgg_execute` **同步**执行（绝不 `vgg_execute_async` — 异步会在未完成的 trace 上抢先出判定）。
- 判定由 frozen `VerdictReport`（`vcli/verdict.py`）承载，**只**从既有 `classify_step_evidence` / `evidence_passed` 构建，**绝不**二次推导（契约：`VerdictReport.from_trace(trace, oracle).verified == evidence_passed(trace, oracle)`）。

`VECTOR_VERDICT` JSON 字段：

| 字段 | 含义 |
|------|------|
| `verified` | bool — `evidence_passed` 的结果（验收唯一真值；`verified == (exit==0)`） |
| `success` | bool — trace 是否成功（步骤都成功，未必有 grounded 证据） |
| `evidence` | `GROUNDED` \| `RAN` \| `FAILED` \| `NO_TRACE`（顶层证据等级） |
| `goal` | 本 turn 的 goal（来自 GoalTree） |
| `n_steps` / `n_grounded` | 步数 / 其中 GROUNDED 的步数 |
| `oracle_names` | 本世界 live verify 命名空间的可调用名（与 GoalVerifier 同源） |
| `per_step` | 每步 `{name, strategy, success, verify, verify_result, evidence}` |
| `error` | 仅 NO_TRACE/错误时填写 |

**退出码契约：** `0` = verified（GROUNDED）·`2` = ran-not-verified（RAN/FAILED）·`1` = error / NO_TRACE（chat/tool_use turn 无确定性 trace → fail-closed）。

**测试驱动（确定性、无网络）：** `VECTOR_FAKE_LLM=<json-path>` 环境变量在**单一** `create_backend` 接缝
(`create_backend_with_fake_seam`) 注入 `tests/harness/fake_backend.py::FakeBackend`，返回一份固定的 decompose 计划。
它**只**替换网络 LLM —— 真实的 decomposer / validator / skill / GoalVerifier / 证据门 / 判定**全部照跑**，所以
`verify='True'` 的假计划仍判 RAN → verified False（接缝绝不绕过任何 verify/permission 层）。未设该变量时，
`create_backend` 行为与生产完全一致。

**PTY 验收 + CI 门：** `tests/harness/pty_cli.py::run_cli_turn` 用 **stdlib `pty`**（不引入 pexpect 依赖）拉起真入口、
读 `VECTOR_VERDICT`、断言 `verified == (exit==0)`。CI 门：`cli_main` + `capability` 两个 pytest marker 已注册
(`pyproject.toml`)；`tests/conftest.py` 的 `pytest_collection_modifyitems` 会**判失败**任何带 `@capability` 却缺
`@cli_main` 的测试（杜绝回退到绕过脚本）。

## ToolHookRegistry — 工具执行钩子

在每个工具执行前后触发回调，用于：
- 自动验证（电机技能后检查位置变化）
- 统计遥测（记录工具调用频率/耗时）
- 链式反应（文件编辑后自动格式化）

```python
class ToolHookRegistry:
    def add_pre_hook(self, hook: Callable) -> None    # 执行前
    def add_post_hook(self, hook: Callable) -> None   # 执行后
    def fire_pre(self, ctx: ToolHookContext) -> None
    def fire_post(self, ctx: ToolHookContext) -> None

@dataclass(frozen=True)
class ToolHookContext:
    tool_name: str
    params: dict
    result: ToolResult | None   # pre-hook 时为 None
    duration: float             # pre-hook 时为 0.0
```

钩子异常被吞掉，不会中断工具执行。

## DynamicSystemPrompt — 动态系统提示

**问题**: System prompt 启动时构建一次，之后机器人状态就过期了。

**解决**: `DynamicSystemPrompt` 是 list 的子类，重写 `__iter__()`。VectorEngine 每次 API 调用都会遍历 system prompt，所以机器人状态每轮都是最新的。

LLM 每次对话都看到：
```
[Robot State]
Position: (10.2, 5.3, 0.28) — hallway
Heading: 23 deg (NNE)
SceneGraph: 8 rooms (6 visited), 9 doors, 12 objects
Exploring: no
Nav stack: running
```

native CLI 另外从 live SceneGraph/`RoomResolver` 重建 `Known rooms: ...`，并携带当前
`known_layout` / `unknown_exploration` 模式；它不从原始 YAML 复制房间坐标。

## RobotContextProvider — 机器人状态采集

从多个来源实时采集状态：

| 字段 | 数据源 | 更新频率 |
|------|--------|---------|
| 位置 (x, y, z) | `base.get_position()` | 每轮对话 |
| 朝向 (度数 + 方位) | `base.get_heading()` | 每轮对话 |
| 当前房间 | `scene_graph.nearest_room()` | 每轮对话 |
| SceneGraph 摘要 | `scene_graph.stats()` + `get_room_summary()` | 每轮对话 |
| 是否在探索 | `explore.is_exploring()` | 每轮对话 |
| 导航栈运行中？ | `explore.is_nav_stack_running()` | 每轮对话 |

优雅降级：没有 base → "No hardware connected"。没有 SceneGraph → 省略房间数据。

## 权限系统

7 层检查（优先级从高到低）：

1. `no_permission` 标志 → 全部放行
2. `deny_tools` 黑名单 → 拒绝
3. `tool.check_permissions()` 返回 deny → 拒绝
4. `session_allow`（用户说了 "always"）→ 放行
5. `is_read_only(params)` → 放行
6. `tool.check_permissions()` 返回 ask → 提示用户确认
7. 默认 → 提示用户确认

电机技能（`navigate_room`、`navigate_xy`、walk、pick）→ 始终 ask。
只读工具（file_read、grep、ros2_topics）→ 始终 allow。

## 完整工具清单 (19 内置 + 臂控技能)

### 内置工具

| 工具 | 类别 | 只读 | 权限 | 说明 |
|------|------|------|------|------|
| file_read | code | 是 | allow | 读取文件（带行号） |
| file_write | code | 否 | ask | 创建/覆盖文件 |
| file_edit | code | 否 | ask | 搜索替换 |
| bash | code | 否 | ask | 执行 shell 命令 |
| glob | code | 是 | allow | 按模式查找文件 |
| grep | code | 是 | allow | 搜索文件内容 |
| web_fetch | general | 是 | allow | 抓取 URL |
| world_query | robot | 是 | allow | 查询世界模型对象 |
| scene_graph_query | robot | 是 | allow | 查询房间/门/物体/路径 |
| ros2_topics | diag | 是 | allow | 列出/hz/echo ROS2 话题 |
| ros2_nodes | diag | 是 | allow | 列出/info ROS2 节点 |
| ros2_log | diag | 是 | allow | 读取机器人日志 |
| nav_state | diag | 是 | allow | 导航/探索状态 |
| terrain_status | diag | 是 | allow | 地形地图文件信息 |
| start_simulation | sim | 否 | ask | 启动 MuJoCo 仿真 |
| stop_simulation | sim | 否 | ask | 停止 MuJoCo 仿真 |
| robot_status | system | 是 | allow | 硬件连接状态 |
| skill_reload | system | 否 | ask | 热加载技能模块 |
| open_foxglove | system | 是 | allow | 打开 Foxglove 可视化 |

### 机械臂技能工具（接入臂控 agent 时动态注册，robot 类别，10 个）

连接 SO-101 arm agent（`vector-cli --sim` 或实体臂）时，以下技能自动包装为 robot 类别工具：

home, wave, scan, detect, describe, pick, place, gripper_open, gripper_close, handover

## CLI 日志分流

交互终端只显示关键进度、压缩后的 action/verdict 和可操作错误，不直接承载网络库、
OpenGL 或 asyncio 的 DEBUG 流。Vector 自身诊断写入私有轮转文件
`~/.vector/logs/vector-cli.log`（默认 INFO，`--verbose` 时 DEBUG，5 MiB × 3 个备份）；
可用 `VECTOR_CLI_LOG_FILE` 覆盖路径。默认受管目录或 CLI 新建的目录使用 0700；当前
日志和轮转代文件始终使用 0600。已有的自定义父目录权限不会被 CLI 擅自修改。第三方
HTTP/SDK wire DEBUG 在 verbose 模式也保持抑制，避免完整 prompt、请求头和代理细节
进入终端或诊断文件。

每次手工动态验收建议创建新的时间戳目录；不要重复使用 `run_01` 覆盖上一轮：

```zsh
RUN_ID="$(date +%Y%m%d_%H%M%S)"
CASE="$PWD/artifacts/benchmarks/afterp2/nav/manual_${RUN_ID}"
install -d -m 700 "$CASE" "$CASE/ros"
git rev-parse HEAD > "$CASE/git-head.txt"
git status --short > "$CASE/git-status.txt"

( umask 077
  VECTOR_CLI_LOG_FILE="$CASE/vector-cli.log" \
  VECTOR_VNAV_LOG_FILE="$CASE/sim.log" \
  ROS_LOG_DIR="$CASE/ros" \
  script -q -f -e -c "vector-cli" "$CASE/cli.typescript"
)
```

进入 CLI 后按正常产品流程启动仿真并逐条发导航命令，结束时输入 `/exit` 或按
`Ctrl-D`。`cli.typescript` 是用户实际看到的 PTY 转录（会保留 ANSI 控制字符），
`vector-cli.log` 是 Vector 诊断日志，`ros/` 是 ROS 日志，`sim.log` 是本轮
bridge/FAR/localPlanner/TARE/RViz 联合日志。用户提供问题复现时，至少同时提交
`cli.typescript`、`vector-cli.log`、`sim.log`、`git-head.txt` 和 RViz 截图；仅有终端
截图通常无法区分拓扑、FAR、localPlanner 与显示残留。

如需跑不经过 LLM 的 P2 产品级验收，应先退出上述手工仿真，避免两套栈争用锁和 ROS
topic，然后执行：

```zsh
python scripts/verify_p2_room_navigation.py --plan-only
python scripts/verify_p2_room_navigation.py
python scripts/verify_p2_room_navigation.py --targets kitchen study guest_bedroom
python scripts/verify_p2_room_navigation.py --all-rooms
```

第二条默认生成时间戳目录 `artifacts/benchmarks/afterp2/nav/dynamic_*`，其中
`report.json` 记录三段真实房间导航、启动/晚加入空 Path、四路路径、registered scan、
localPlanner 参与和终态清理证据，`sim.log` 保留对应运行日志。2026-07-29 的最终
带 Piper 验收证据为 `dynamic_codex_20260729_14`，三段均成功，反向段还覆盖了
localPlanner 先横移再进直连门的有效局部轨迹；这项证据证明
localPlanner 在真实链路中参与，不等同于受控移动障碍专项认证。`--targets` 可按顺序
验收任意房间/别名并核对多门链；`--all-rooms` 使用最短的相邻腿序列覆盖 8 个房间和
9 扇门。每腿完成后会写入 `progress.json`，长验收即使中断也能保留已完成 case。
2026-07-30 的其余房间证据位于
`dynamic_other_rooms_codex_20260730_03_master`、
`dynamic_other_rooms_codex_20260730_04_kitchen_study_guest` 和
`dynamic_other_rooms_codex_20260730_05_bathroom`，五个此前未测试的目标房间均成功；
`dynamic_other_rooms_codex_20260730_06_remaining_doors` 又补齐
`study-hallway` 与 `living_room-hallway`。聚合证据动态覆盖 8/8 房间和 9/9 物理门。

## Session 持久化

JSONL 格式，原子写入 + fsync：
```
{"type":"user","content":"去厨房","ts":"..."}
{"type":"assistant","text":"","tool_use":[{"name":"navigate_room","input":{"room":"厨房"}}],"ts":"..."}
{"type":"tool_result","results":[{"content":"NavigateSkill succeeded: canonical_room=kitchen..."}],"ts":"..."}
{"type":"assistant","text":"","tool_use":[{"name":"verify","input":{"expr":"in_room('kitchen')"}}],"ts":"..."}
{"type":"assistant","text":"到了厨房，你要我看看有什么吗？","ts":"..."}
```

50 条记录自动压缩，防止上下文溢出。

## 探索事件流

探索期间，房间发现事件实时显示在 CLI：

```
vector> explore
  start_simulation(sim_type="go2") ... ok 2.1s
  explore() ... ok
  Entered hallway (1/8)
  Entered kitchen (2/8)
  Entered dining_room (3/8)
  ...
  Exploration finished — 8 rooms
```

由 `explore.py` 的 `set_event_callback()` 驱动，在 `vcli/cli.py` 启动时接入。

## 文件目录

```
vcli/
├── cli.py                  # 入口、REPL 循环、斜杠命令
├── engine.py               # VectorEngine — 多轮 tool_use agent 循环
├── intent_router.py        # IntentRouter — 意图路由（关键词 → 类别）
├── hooks.py                # ToolHookRegistry — 工具执行钩子
├── prompt.py               # 系统提示构建器（静态 + 动态块）
├── robot_context.py        # RobotContextProvider（实时机器人状态）
├── dynamic_prompt.py       # DynamicSystemPrompt（每轮自动刷新）
├── session.py              # JSONL session 持久化
├── config.py               # ~/.vector/config.yaml 加载器
├── permissions.py          # 7层权限检查器
├── backends/
│   ├── __init__.py         # LLMBackend Protocol + create_backend 工厂
│   ├── anthropic.py        # Anthropic Messages API（流式）
│   └── openai_compat.py    # OpenRouter / Ollama / vLLM
├── cognitive/
│   ├── vgg_harness.py        # VGG 主循环（复杂 NL 任务分解入口）
│   ├── goal_decomposer.py    # NL → 子步骤计划
│   ├── goal_executor.py      # 执行计划
│   ├── goal_verifier.py      # 验证步骤结果
│   ├── strategy_selector.py  # 策略选择
│   ├── capabilities/         # 能力映射
│   └── worlds/               # 世界模型适配
└── tools/
    ├── base.py             # Tool Protocol, @tool 装饰器,
    │                       # ToolRegistry, CategorizedToolRegistry
    ├── __init__.py         # discover_all_tools(), discover_categorized_tools()
    ├── file_tools.py       # file_read, file_write, file_edit
    ├── bash_tool.py        # bash
    ├── search_tools.py     # glob, grep
    ├── robot.py            # world_query, robot_status
    ├── sim_tool.py         # start_simulation, stop_simulation
    ├── web_tool.py         # web_fetch
    ├── skill_wrapper.py    # SkillWrapperTool + wrap_skills() + 恢复提示
    ├── scene_graph_tool.py # scene_graph_query（7种查询）
    ├── ros2_tools.py       # ros2_topics, ros2_nodes, ros2_log
    ├── nav_tools.py        # nav_state, terrain_status
    ├── reload_tool.py      # skill_reload（热加载）
    └── foxglove_tool.py    # open_foxglove
```
