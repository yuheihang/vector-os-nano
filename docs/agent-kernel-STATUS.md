# Vector OS — STATUS (resume anchor)

One-page "where are we / what's next". Read this first; the GOAL is in [../CLAUDE.md](../CLAUDE.md)
→ North Star; durable design is [ARCHITECTURE.md](ARCHITECTURE.md); decision history is
[DECISIONS.md](DECISIONS.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md). Per-round
narrative + the campaign plan live in `~/.vector-nano-loop/{journal,campaign}.md`.

updated: 2026-07-29 · P1/P2 dynamic GUI acceptance is green, including the CLI follow-up: native Go2+Piper startup now requires live bridge/odom/joint state, RViz uses only installed core plugins, GUI/headless camera rendering selects the correct GL backend, interactive logs are separated from rotating diagnostics, and coordinate navigation cannot livelock on mismatched arrival thresholds. The older R42 nav+grasp campaign below remains historical context; P1/P2 additively extend the verify spine without weakening the evidence gate.
goal:    agent-orchestration runtime for physical AI — plan · route to the right MODEL/skill ·
         verify each step · recover. Sim-first; bare `vector-cli` + NL is the only acceptance interface.
         CURRENT THRUST: prove the 3 under-proven North-Star axes (route-to-MODEL ✓ now at the ORCHESTRATION layer · cross-embodiment · live orchestration), using the moat to grade each.
phase:   M2 cross-model — a learned detector is the first real 2nd model family, now routed-to BY THE PRODUCER via the
         engine capability-dispatch path (D50), not only inside the grasp skill (D48); colour selection PERCEPTUAL (D49).
owns:    perception/grounding_dino.py, perception/detector_capability.py (now AGENT-bound, lazy cold-turn rebind),
         perception/go2_grasp_perception.py (detect→gdino), skills/perception_grasp.py (named→detector routing +
         CONSUMES a producer box), worlds/robot.py (register_capabilities binds the agent for the rebind).
         (At that historical R39 checkpoint the moat was byte-unchanged across 50 decisions; P1 later extends it
         additively with named-room verification and goal-scoped navigation causation.)
doing:   R39 — nav+grasp now COMPLETES end-to-end and GROUNDS the right object intermittently (honest, not a landed
         headline). ROOT CAUSE found via Debug Protocol (NOT the A/B-probe "off-axis lateral IK" theory): the grasp
         never grounded because PERCEPTION was silently degraded. Two real, independent fixes (both verified):
         (1) `timm` — ALREADY declared in pyproject (perception extra) but MISSING from .venv → EdgeTAM failed to LOAD →
             coarse box-rect mask → depth centroid averaged can+table → z collapsed to ~0.13 → gripper closed below the
             can → no weld. Synced timm==1.0.27 into .venv (env-sync of a declared dep, NOT a new dependency / not a gate).
         (2) EdgeTAM scores-shape bug (perception/tracker.py:330): transformers>=5 returns object_score_logits (N,1), so
             `float(scores[i])` on a (1,)-array raised TypeError → segment() fell back to box-rect EVERY time even with
             timm present. Flatten scores to 1-D (commit 8f9851e). Verified in isolation: EdgeTAM now returns a real mask.
         RESULT (real sim, decompose→vgg_execute, retries=0, 3 trials each): GREEN grasp_GROUNDED 1/3 (green t1: real
         perceive 2.3cm @ y=3.00 z=0.322 → weld + shoulder lift + holding_object('pickable_bottle_green') TRUE); RED 0/3.
         The chain mechanically completes (FAR→dock converges +X→perceive→_approach_object vy-track→PickTopDown weld→lift).
         At that R39 checkpoint the spine was byte-unchanged (verified empty diff 7b220d9..HEAD); P1 later makes the
         additive changes summarized above. Probes: scripts/probe_r39_e2e_green_red.py
         (acceptance), probe_r39_reperceive_after_seat.py (proved: do NOT re-perceive after approach — close framing looks
         OVER the table → garbage), probe_r39_debug_floor_vs_cans.py. Artifacts /tmp/r39_e2e/*.png + e2e.json.

blocked: NOT a CEO gate — a perception-quality round. The remaining miss is DETECTION SELECTION at the dock framing:
         the 3 pickable objects (blue y=2.78, green 3.00, red 3.22) are only 22cm apart and the dog perceives them ~0.85m
         away and slightly off-center (the dock leaves a small per-trial pose residual), so grounding-dino's colour
         grounding intermittently picks a NEIGHBOUR's box and the back-projected z sometimes still lands low (table). Per
         trial the perceived xy tracks the dog's dock-residual: green t2 grabbed RED's y, red t1 grabbed BLUE's y, etc.
         KNOWN (spine, do-not-touch): re-plan still drops strategy_params (empty query on retry) — retries pinned to 0.
next:    R40 — perception RELIABILITY at the dock framing (NON-gated, non-spine), to lift GREEN→~3/3 and land RED honestly:
         (a) tighten the dock so the perceive pose is repeatable head-on AND closer-but-not-over-table (a fixed perceive
             standoff where the 3 cans subtend more pixels); (b) constrain detection to the near-table depth band + select
             the box nearest the commanded colour's expected screen region, reject low-z (table) back-projections FAIL-LOUD;
             (c) consider a colour-segmentation cross-check (front_object colour resolver) to disambiguate the 3 close cans.
         Then bare-cli two-turn "启动 go2 带机械臂 → 去桌子那里把绿色的瓶子拿起来" → nav RAN + grasp GROUNDED end-to-end.
         Gated leaps (CEO queue): re-plan strategy_params-preservation (SPINE — D52), cross-EMBODIMENT (g1), explore
         (TARE), VLN (SysNav), merge→master.
         ALSO record for reproducibility: `.venv` must have `timm` (uv pip install 'timm>=1.0'); EdgeTAM backbone
         repvit_m1.dist_in1k fetches from HF on first load (network needed once, then cached).
         Bare vector-cli + NL = ONLY acceptance; spine only STRICTER; never trust skill.success / sub-agent claims.


## Standing facts (durable)
- **Branch `feat/orchestrator-redesign`** off master; `feat/playground-vln` is ABANDONED (never touch/delete).
- **Honest-verify axis** (the moat's core): a step grades GROUNDED only when a deterministic predicate
  reads an oracle the ACTOR cannot author (actor-causation + structural classifier). The sandbox may only get
  STRICTER (rule 5). P1 adds `in_room` to the grounded predicate set and extends actor capture to the
  goal-scoped bridge route; no evidence shortcut or robot bypass was introduced.
- **Cross-MODEL seam (D48):** engine.py builds a CapabilityRegistry, calls world.register_capabilities, threads
  names→StrategySelector + registry→GoalExecutor. A world registers a Capability(kind=chat|detector|planner|vla|…);
  the spine grades it, it never self-certifies. First real entry: the grounding-dino `detect` capability.
- **Acceptance = bare `vector-cli` + NL only** (cli.main PTY asserting the verify VERDICT); `VECTOR_FAKE_LLM`
  fakes ONLY the network LLM. PTY harness needs HF_HOME pinned for the offline detector (D48 note).
- **Native named navigation (P1).** Bare `vector-cli` exposes two unambiguous motor tools:
  `navigate_room(room)` for semantic destinations and `navigate_xy(x,y)` for explicit coordinates.
  Both delegate to the formal `NavigateSkill`; named-room aliases, availability, and centres are
  single-sourced by `RoomResolver` from the live SceneGraph. The native prompt reads the same room
  vocabulary: `known_layout` may expose every prior room loaded into that graph, while
  `unknown_exploration` exposes only rooms discovered online. An unknown name fails closed and never
  falls back to an LLM-invented coordinate. Known-layout startup treats `room_layout.yaml` geometry as
  authoritative over stale persisted centres/doors while preserving objects, descriptions, and visits.
- **Navigation verify + causation (P1).** Room goals use the deterministic `in_room(room_id)`
  predicate (geometry first, with an observable `nearest_center` downgrade); coordinate goals use
  `at_position`. Each navigation carries a `goal_id`, and actor causation is attributed only from
  actual non-zero path-following `cmd_vel` accepted for that same goal plus measured base motion.
  Merely publishing a goal, unrelated drift, or commands belonging to another goal cannot count as
  CAUSED. Real commanded movement that misses the post-condition is RAN; only a passing post-condition
  reaches GROUNDED.
- **Coordinate convergence + bounded recovery (P1).** `navigate_xy` and `at_position` share the
  same 0.5 m acceptance radius at the transport and verification boundaries. Slow but cumulative
  progress resets the configured stall watchdog; a stationary robot still fails closed. An identical
  failed navigation/post-condition pair is attempted at most three times, then returns one compact
  failure summary instead of consuming the native turn budget.
- **Safe known-layout navigation (P2).** `room_layout.yaml` schema v2 supplies room polygons and
  trusted door geometry (width, normal, two oriented standoffs). `SceneGraph.plan_door_route`
  rejects invalid/untrusted/narrow edges and expands every crossing into structured
  `door_pre/door_center/door_post` waypoints plus the room goal. `NavigateSkill` submits those
  points to FAR one segment at a time with no legacy fallback. Each motor policy carries a unique
  `segment_id` and must be ACKed by the bridge; failed/no-path/out-of-tolerance segments disarm the
  nav gate and hold zero. Door segments retain no-reverse, topology, tolerance, ACK, and obstacle
  safety but use the path follower's normal adaptive speed rather than a fixed low-speed cap
  (owner override 2026-07-29). RViz separately labels the door topology, FAR global path,
  localPlanner path, executed odometry path, and current red goal.
- **Arm kinematics ownership.** Piper has always used its own 6-DoF MuJoCo Jacobian IK; SO-101 uses
  its separate 5-DoF Pinocchio solver. The generic Agent now constructs SO-101 IK only for arms
  exposing `set_ik_solver` and synchronizes EE state through the arm adapter's own `fk()` first.
- **Dynamic startup/RViz acceptance (P1/P2).** `start_simulation` reports success only after the
  complete control plane is usable: the managed subprocess is alive, DDS endpoints are matched,
  odometry sustains at least 5 Hz, FAR has published a stable non-empty `global_vertex` V-Graph,
  and requested Piper joint state has arrived. Every run owns an isolated ROS domain and session
  paths; stop/restart releases the old rclpy context before selecting a new domain. The launcher
  continuously monitors bridge/localPlanner/FAR/TARE/terrain processes after Ready, and an early
  death returns a compact root cause plus the retained per-session log. Terrain persistence loads
  into the live accumulator and merges atomically without shrinking the canonical seed.
  `launch_explore.sh` uses EGL device 0 after single-GPU remapping, selects GLFW for GUI camera
  rendering and EGL headless, and cleans up idempotently. The portable RViz config has no unbuilt
  FAR demo panel/tool dependencies and uses the published `vehicle` TF.
- **CLI observability split.** The interactive terminal is a product surface: it shows only concise
  progress, compact action/verdict rows, and actionable errors. Vector diagnostics go to a private
  rotating file (`~/.vector/logs/vector-cli.log`; DEBUG with `--verbose`, INFO otherwise).
  Third-party HTTP/SDK wire DEBUG remains suppressed so prompts and headers are not copied into logs.

## Pending CEO gates (decision queue — do NOT cross autonomously)
- **DEP `timm>=1.0` (1.0.27) — CEO-APPROVED 2026-06-23.** Added to pyproject + .venv to make EdgeTAM actually LOAD
  (its undeclared backbone; EdgeTAM never loaded across the grasp campaign D17-D51 → masks were box-rect). Standard
  PyTorch-image-models lib; EdgeTAM backbone repvit_m1.dist_in1k fetches from HF once then caches. No longer a gate.
- Merge/release `feat/orchestrator-redesign` → master.
- cross-EMBODIMENT (g1: removed, zero python — large rebuild) ; explore→TARE + remaining nav-stack
  colcon bring-up (DQ-15) ; VLN→SysNav venv (DQ-16). New external deps / new-or-changed interfaces /
  hardware / security. Real SO-101 arm acceptance gated on `ls /dev/ttyACM*` (absent — sim only).
