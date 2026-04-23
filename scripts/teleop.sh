#!/bin/bash
LEADER_PORT=can3
FOLLOWER_PORT=can2

lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id=follower \
  --teleop.type=piper_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id=leader