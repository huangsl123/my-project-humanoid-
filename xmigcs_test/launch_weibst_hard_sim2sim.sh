#!/usr/bin/env bash
set -euo pipefail

sim_root="/home/eai/my project/xSIM_MUJOCO"
control_root="/home/eai/my project/xmigcs_test"
deploy_python="/home/eai/Tools/miniforge3/envs/xmimic_deploy/bin/python"
overlay_root="$control_root/.ros-overlay/opt/ros/humble"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export AMENT_PREFIX_PATH="$overlay_root:/opt/ros/humble${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:$overlay_root/local/lib/python3.10/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu:$overlay_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

model="$control_root/policy/beyond_mimic/model/weibst_hard_best.onnx"
if [[ ! -s "$model" ]]; then
  echo "Validated Weibst ONNX is not installed yet: $model" >&2
  exit 1
fi

sim_pid=""
cleanup() {
  if [[ -n "$sim_pid" ]] && kill -0 "$sim_pid" 2>/dev/null; then
    kill "$sim_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "Starting xSIM MuJoCo (evt2)..."
cd "$sim_root"
"$deploy_python" scripts/simulator_view_asyn.py -m evt2 &
sim_pid=$!

sleep 3
if ! kill -0 "$sim_pid" 2>/dev/null; then
  wait "$sim_pid"
  exit 1
fi

echo "Starting xMIGCS. In its terminal press 'u' to select weibst_hard."
echo "The MuJoCo window must publish /base/odometry before policy entry."
cd "$control_root"
"$deploy_python" rl_control_node.py
