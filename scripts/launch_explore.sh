#!/bin/bash
# Go2 + Vector Nav Stack + TARE Autonomous Exploration — one-command launch
#
# Usage:
#   cd /media/fishyu/fish-14tb-2/YuXi/go2armagent/vector-os-nano
#   ./scripts/launch_explore.sh              # MuJoCo viewer + RViz
#   ./scripts/launch_explore.sh --no-gui     # headless MuJoCo
#
# Go2 will autonomously explore the environment using TARE planner.
# TARE finds frontiers, plans TSP tours, and publishes /way_point goals.
# FAR planner routes to each waypoint. localPlanner handles obstacles.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Session-scoped files are supplied by SimStartTool.  The fallbacks preserve
# direct/manual launcher compatibility without forcing Python orchestration.
NAV_ACTIVE_FILE="${VECTOR_NAV_ACTIVE_FILE:-/tmp/vector_nav_active}"
NAV_STALLED_FILE="${VECTOR_NAV_STALLED_FILE:-/tmp/vector_nav_stalled}"
NAV_RESET_FILE="${VECTOR_NAV_RESET_FILE:-/tmp/vector_reset_pose}"
NAV_REPLAY_FILE="${VECTOR_NAV_REPLAY_FILE:-/tmp/vector_terrain_replay}"

# Hold one process-wide lock for the complete navigation stack lifetime.
# A stale lock file is harmless; flock tracks the live file description.  If
# an old CLI crashed but its ROS children survived, they keep the lock and a
# second stack fails loudly instead of mixing old goals into a new RViz.
VNAV_LOCK_FILE="${VECTOR_VNAV_LOCK_FILE:-/tmp/vector_vnav.lock}"
exec 9>"$VNAV_LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another Vector navigation stack is still running." >&2
    echo "Stop the old simulation before starting a new one." >&2
    exit 73
fi

# Main project directories
GO2ARM_ROOT="${GO2ARM_ROOT:-/media/fishyu/fish-14tb-2/YuXi/go2armagent}"
NAV_STACK="${NAV_STACK:-$GO2ARM_ROOT/vector_navigation_stack}"
TARE_ROOT="${TARE_ROOT:-$GO2ARM_ROOT/third_party/tare_planner}"

NO_GUI=""
for arg in "$@"; do
    case $arg in --no-gui) NO_GUI="--no-gui" ;; esac
done

# MuJoCo's GUI viewer owns a GLFW context. Camera rendering must use the same
# backend in GUI mode; forcing EGL there can fail to make its context current
# and leaves RViz's Image display empty. Headless mode keeps EGL.
if [ -z "$NO_GUI" ]; then
    export MUJOCO_GL="${VECTOR_MUJOCO_GUI_GL:-glfw}"
else
    export MUJOCO_GL="${VECTOR_MUJOCO_HEADLESS_GL:-egl}"
fi

# Local VLM (Ollama) — set if not already in environment
export VECTOR_VLM_URL="${VECTOR_VLM_URL:-http://localhost:11434/v1}"
export VECTOR_VLM_MODEL="${VECTOR_VLM_MODEL:-gemma4:e4b}"

# Python packages such as convex_mpc are installed in the active Conda
# environment, so only add this repository itself to PYTHONPATH.
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# TARE uses its bundled OR-Tools shared libraries.
export LD_LIBRARY_PATH="$TARE_ROOT/src/tare_planner/or-tools/lib:${LD_LIBRARY_PATH:-}"

export ROBOT_CONFIG_PATH="unitree/unitree_go2"

# CUDA_VISIBLE_DEVICES renumbers a single selected physical GPU to local index
# zero. Normalise stale activation values (for example physical index "2") so
# MuJoCo camera rendering does not fail and leave RViz's Image display empty.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "$CUDA_VISIBLE_DEVICES" != *,* ]]; then
    export MUJOCO_EGL_DEVICE_ID=0
else
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
fi

# ROS 2 Humble and the compiled navigation workspace
source /opt/ros/humble/setup.bash
source "$NAV_STACK/install/setup.bash"

PIDS=()
CLEANING_UP=0
PARENT_WATCH_PID=""
require_alive() {
    local pid="$1"
    local name="$2"
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    set +e
    wait "$pid"
    local status=$?
    set -e
    if [ "$status" -eq 0 ]; then
        status=1
    fi
    echo "ERROR: $name exited during the navigation session (status $status)." >&2
    return "$status"
}

process_start_ticks() {
    # Field 22 in /proc/<pid>/stat is the process start time.  Pairing it with
    # the PID prevents an extremely rare PID reuse from making a dead planner
    # look healthy.  Strip through the final ") " first because the comm field
    # is parenthesised and may itself contain spaces.
    local pid="$1"
    local stat_line
    local stat_tail

    IFS= read -r stat_line 2>/dev/null < "/proc/$pid/stat" || return 1
    stat_tail="${stat_line##*) }"
    set -- $stat_tail
    if [ "$#" -lt 20 ]; then
        return 1
    fi
    case "$1" in
        Z|X|x) return 1 ;;
    esac
    printf '%s\n' "${20}"
}

find_descendant_process() {
    # ROS 2 launch keeps running while any included process (for example
    # graph_decoder) is alive.  Locate the real planner below that launch
    # process by its kernel comm name rather than using a global pgrep, which
    # could accidentally bind to another ROS_DOMAIN_ID/session.
    local parent_pid="$1"
    local expected_comm="$2"
    local child_pid
    local child_comm
    local nested_pid

    while IFS= read -r child_pid; do
        if [ -z "$child_pid" ]; then
            continue
        fi
        child_comm=""
        IFS= read -r child_comm 2>/dev/null < "/proc/$child_pid/comm" || true
        if [ "$child_comm" = "$expected_comm" ]; then
            printf '%s\n' "$child_pid"
            return 0
        fi
        if nested_pid="$(find_descendant_process "$child_pid" "$expected_comm")"; then
            printf '%s\n' "$nested_pid"
            return 0
        fi
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    return 1
}

wait_for_descendant_process() {
    local launch_pid="$1"
    local expected_comm="$2"
    local timeout_s="$3"
    local deadline=$((SECONDS + timeout_s))
    local process_pid
    local start_ticks

    while kill -0 "$launch_pid" 2>/dev/null; do
        process_pid=""
        if process_pid="$(find_descendant_process "$launch_pid" "$expected_comm")"; then
            start_ticks=""
            if start_ticks="$(process_start_ticks "$process_pid")"; then
                printf '%s %s\n' "$process_pid" "$start_ticks"
                return 0
            fi
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ERROR: $expected_comm did not start within ${timeout_s}s." >&2
            return 1
        fi
        sleep 0.1
    done

    echo "ERROR: FAR planner launch exited before $expected_comm started." >&2
    return 1
}

require_process_identity_alive() {
    local pid="$1"
    local expected_start_ticks="$2"
    local name="$3"
    local current_start_ticks

    current_start_ticks=""
    if current_start_ticks="$(process_start_ticks "$pid")" &&
       [ "$current_start_ticks" = "$expected_start_ticks" ]; then
        return 0
    fi

    echo "ERROR: $name exited during the navigation session." >&2
    return 1
}

monitor_critical_processes() {
    # The launcher is the health boundary observed by SimStartTool.  Keep
    # checking every process that is required for navigation after the Ready
    # marker too; otherwise a dead FAR/local/TARE process leaves the bridge
    # alive and makes the next command look like a mysterious planning timeout.
    while true; do
        require_alive "$BRIDGE_PID" "Go2 navigation bridge"
        require_alive "$LOCAL_PLANNER_PID" "localPlanner"
        require_alive "$VEHICLE_TF_PID" "sensor-to-vehicle transform"
        require_alive "$CAMERA_TF_PID" "sensor-to-camera transform"
        require_alive "$SENSOR_SCAN_PID" "sensorScanGeneration"
        require_alive "$TERRAIN_PID" "terrainAnalysis"
        require_alive "$TERRAIN_EXT_PID" "terrainAnalysisExt"
        require_process_identity_alive \
            "$FAR_PLANNER_PID" "$FAR_PLANNER_START_TICKS" "far_planner"
        require_alive "$FAR_LAUNCH_PID" "FAR planner launch"
        require_alive "$TARE_LAUNCH_PID" "TARE planner launch"
        sleep 1
    done
}

signal_tree() {
    local parent_pid="$1"
    local signal_name="$2"
    local child_pid
    while read -r child_pid; do
        if [ -n "$child_pid" ]; then
            signal_tree "$child_pid" "$signal_name"
        fi
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    kill -s "$signal_name" "$parent_pid" 2>/dev/null || true
}

cleanup() {
    if [ "$CLEANING_UP" -eq 1 ]; then
        return
    fi
    CLEANING_UP=1
    trap - EXIT INT TERM
    echo ""
    echo "Stopping all processes..."
    if [ -n "$PARENT_WATCH_PID" ]; then
        kill "$PARENT_WATCH_PID" 2>/dev/null || true
    fi
    for p in "${PIDS[@]}"; do
        signal_tree "$p" TERM
    done
    sleep 1
    for p in "${PIDS[@]}"; do
        if kill -0 "$p" 2>/dev/null; then
            signal_tree "$p" KILL
        fi
    done
    rm -f "$NAV_ACTIVE_FILE" "$NAV_STALLED_FILE" \
          "$NAV_RESET_FILE" "$NAV_REPLAY_FILE" 2>/dev/null
    wait 2>/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

# When launched by vector-cli, watch its exact PID.  A normal SimStop sends
# SIGTERM to this complete Popen-created process group.  This watcher covers an
# abnormal parent exit too, so no bridge/planner can retain DDS endpoints or a
# log file descriptor into the next session.
if [ -n "${VECTOR_VNAV_PARENT_PID:-}" ]; then
    if ! [[ "$VECTOR_VNAV_PARENT_PID" =~ ^[0-9]+$ ]]; then
        echo "ERROR: VECTOR_VNAV_PARENT_PID must be numeric." >&2
        exit 64
    fi
    (
        while kill -0 "$VECTOR_VNAV_PARENT_PID" 2>/dev/null; do
            sleep 1
        done
        echo "ERROR: vector-cli parent exited; stopping navigation stack." >&2
        if [ "${VECTOR_VNAV_MANAGED_SESSION:-0}" = "1" ]; then
            kill -TERM -- "-$$" 2>/dev/null || true
        else
            kill -TERM "$$" 2>/dev/null || true
        fi
    ) &
    PARENT_WATCH_PID=$!
fi

RVIZ_CFG="$REPO_DIR/config/vnav.rviz"

# Clean only this session's control flags (prevents unwanted startup movement).
rm -f "$NAV_ACTIVE_FILE" "$NAV_STALLED_FILE" \
      "$NAV_RESET_FILE" "$NAV_REPLAY_FILE" 2>/dev/null

echo "============================================"
echo "  Go2 + TARE Autonomous Exploration"
echo "============================================"
echo "  MuJoCo: Go2 MPC in house scene"
echo "  Local:  localPlanner + pathFollower"
echo "  Global: FAR Planner (visibility graph)"
echo "  Explore: TARE Planner (frontier TSP)"
echo "  Terrain: terrainAnalysis + ext"
echo "============================================"

# 1. Bridge (MuJoCoGo2 → ROS2 topics)
echo "[1/8] Starting bridge..."
python3 "$SCRIPT_DIR/go2_vnav_bridge.py" $NO_GUI &
BRIDGE_PID=$!
PIDS+=("$BRIDGE_PID")
sleep 7
require_alive "$BRIDGE_PID" "Go2 navigation bridge"

# 2. Local planner stack.  Start localPlanner directly so the checked-in Go2
# parameter file is the actual runtime configuration.  The package's bundled
# launch file hard-codes a 0.6 x 0.6 m footprint, adjacentRange=4.25 and
# maxSpeed=2.0, silently ignoring config/local_planner_go2.yaml.
echo "[2/8] Starting local planner..."
LOCAL_PLANNER_SHARE="$(ros2 pkg prefix local_planner)/share/local_planner"
LOCAL_PLANNER_CONFIG="$REPO_DIR/config/local_planner_go2.yaml"
ros2 run local_planner localPlanner --ros-args \
    --params-file "$LOCAL_PLANNER_CONFIG" \
    -p pathFolder:="$LOCAL_PLANNER_SHARE/paths" &
LOCAL_PLANNER_PID=$!
PIDS+=("$LOCAL_PLANNER_PID")

# Preserve the transforms previously supplied by local_planner.launch.  The
# Python follower consumes /path itself, so the disabled C++ pathFollower node
# is intentionally not launched.
ros2 run tf2_ros static_transform_publisher \
    --x -0.3 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 \
    --frame-id sensor --child-frame-id vehicle &
VEHICLE_TF_PID=$!
PIDS+=("$VEHICLE_TF_PID")
ros2 run tf2_ros static_transform_publisher \
    --x 0 --y 0 --z 0 --yaw -1.5707963 --pitch 0 --roll -1.5707963 \
    --frame-id sensor --child-frame-id camera &
CAMERA_TF_PID=$!
PIDS+=("$CAMERA_TF_PID")
sleep 4
require_alive "$LOCAL_PLANNER_PID" "localPlanner"
require_alive "$VEHICLE_TF_PID" "sensor-to-vehicle transform"
require_alive "$CAMERA_TF_PID" "sensor-to-camera transform"

# 3. Sensor scan generation (produces /state_estimation_at_scan for TARE)
echo "[3/8] Starting sensor scan generation..."
ros2 run sensor_scan_generation sensorScanGeneration &
SENSOR_SCAN_PID=$!
PIDS+=("$SENSOR_SCAN_PID")
sleep 1

# 4. Terrain analysis
echo "[4/8] Starting terrain analysis..."
ros2 run terrain_analysis terrainAnalysis --ros-args \
  -p clearDyObs:=true \
  -p minDyObsDis:=0.14 \
  -p minOutOfFovPointNum:=20 \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 \
  -p limitGroundLift:=true \
  -p maxGroundLift:=0.05 \
  -p minDyObsVFOV:=-30.0 \
  -p maxDyObsVFOV:=35.0 &
TERRAIN_PID=$!
PIDS+=("$TERRAIN_PID")
ros2 run terrain_analysis_ext terrainAnalysisExt --ros-args \
  -p obstacleHeightThre:=0.15 \
  -p maxRelZ:=1.5 &
TERRAIN_EXT_PID=$!
PIDS+=("$TERRAIN_EXT_PID")
sleep 3

# 5. FAR Planner (routes to TARE waypoints)
# Deploy the Go2-tuned FAR configuration before launching FAR.
FAR_CONFIG_DIR="$NAV_STACK/install/far_planner/share/far_planner/config"

if [ ! -d "$FAR_CONFIG_DIR" ]; then
    echo "ERROR: FAR config directory not found:"
    echo "  $FAR_CONFIG_DIR"
    echo "Build vector_navigation_stack first."
    exit 1
fi

cp "$REPO_DIR/config/far_go2_indoor.yaml" \
   "$FAR_CONFIG_DIR/indoor.yaml"

echo "[5/8] Starting FAR planner..."
ros2 launch far_planner far_planner.launch.py \
    config:=indoor \
    rviz:=false &
FAR_LAUNCH_PID=$!
PIDS+=("$FAR_LAUNCH_PID")
sleep 3
FAR_PLANNER_IDENTITY="$(
    wait_for_descendant_process "$FAR_LAUNCH_PID" "far_planner" 10
)"
read -r FAR_PLANNER_PID FAR_PLANNER_START_TICKS <<< "$FAR_PLANNER_IDENTITY"
require_process_identity_alive \
    "$FAR_PLANNER_PID" "$FAR_PLANNER_START_TICKS" "far_planner"

# 6. TARE Planner (autonomous exploration — includes navigationBoundary)
# Deploy Go2-tuned config BEFORE launch (kAutoStart=false, tuned margins)
cp "$REPO_DIR/config/tare_go2_indoor.yaml" \
   "$NAV_STACK/install/tare_planner/share/tare_planner/indoor_small.yaml" 2>/dev/null
echo "[6/7] Starting TARE exploration planner..."
ros2 launch tare_planner explore.launch.py \
    scenario:=indoor_small \
    rviz:=false &
TARE_LAUNCH_PID=$!
PIDS+=("$TARE_LAUNCH_PID")
sleep 2

# 7. Visualization + optional RViz
echo "[7/7] Starting visualization tools..."
ros2 run visualization_tools visualizationTools 2>/dev/null &
VISUALIZATION_PID=$!
PIDS+=("$VISUALIZATION_PID")

if [ -z "$NO_GUI" ]; then
    echo "Starting RViz with: $RVIZ_CFG"
    rviz2 -d "$RVIZ_CFG" &
    PIDS+=($!)
else
    echo "Headless mode: RViz disabled."
fi

# No seed movement, no nav flag. Dog stays still until user gives a command.
# TARE has kAutoStart=false — waits for /start_exploration from ExploreSkill.
# Nav flag created by ExploreSkill or NavigateSkill when needed.
require_alive "$BRIDGE_PID" "Go2 navigation bridge"
require_alive "$LOCAL_PLANNER_PID" "localPlanner"
require_alive "$VEHICLE_TF_PID" "sensor-to-vehicle transform"
require_alive "$CAMERA_TF_PID" "sensor-to-camera transform"
require_alive "$SENSOR_SCAN_PID" "sensorScanGeneration"
require_alive "$TERRAIN_PID" "terrainAnalysis"
require_alive "$TERRAIN_EXT_PID" "terrainAnalysisExt"
require_process_identity_alive \
    "$FAR_PLANNER_PID" "$FAR_PLANNER_START_TICKS" "far_planner"
require_alive "$FAR_LAUNCH_PID" "FAR planner launch"
require_alive "$TARE_LAUNCH_PID" "TARE planner launch"

echo ""
echo "Ready! Dog is standing still."
echo "  Use vector-cli to control:"
echo "    explore     — start autonomous exploration"
echo "    go to X     — navigate to a room"
echo "    stop        — halt all movement"
echo "  Ctrl+C to shut down."
echo ""

monitor_critical_processes
