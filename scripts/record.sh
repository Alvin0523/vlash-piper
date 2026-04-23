#!/bin/bash
LEADER_PORT=can3
FOLLOWER_PORT=can2

OVERHEAD_SERIAL=048322071496  # USB 3.2 camera
WRIST_SERIAL=348122073292     # USB 2.1 camera

DATASET_NAME="wm/piper-task1"
NUM_EPISODES=20
TASK_DESC="Pick up the object"

CAMERAS="{
  wrist: {type: intelrealsense, serial_number_or_name: $WRIST_SERIAL, width: 640, height: 480, fps: 15},
  overhead: {type: intelrealsense, serial_number_or_name: $OVERHEAD_SERIAL, width: 640, height: 480, fps: 30}
}"

lerobot-record \
  --robot.type=piper_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower \
  --robot.cameras="$CAMERAS" \
  --teleop.type=piper_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=leader \
  --dataset.repo_id="$DATASET_NAME" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.single_task="$TASK_DESC" \
  --dataset.push_to_hub=false \
  --dataset.root=./data \
  --display_data=true