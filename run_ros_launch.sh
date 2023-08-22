#!/bin/bash

# 设置ROS环境变量
source /opt/ros/melodic/setup.bash
source /home/sheldon/mybot_ws_camera/devel/setup.bash

# 运行roslaunch命令
roslaunch mybot_gazebo t4_worldV.launch

rosrun active_vision record_images.py
