# -*- coding: utf-8 -*-
import os
import time

yaw = 0.0

# 文件路径
file_path = "/home/sheldon/mybot_ws_camera/src/mybot_gazebo/launch/t4bot_robotV.launch"  # 将文件路径替换为实际的文件路径

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 修改第17行的内容
line_number_to_modify = 16  # 行号从0开始计数，所以要修改第17行，行号为16
new_line_content = '  <arg name="yaw" default="' + str(yaw) + '"/>\n'  # 修改后的行内容

yaw += 0.1

# 替换指定行的内容
lines[line_number_to_modify] = new_line_content

# 将修改后的内容写回文件
with open(file_path, 'w') as file:
    file.writelines(lines)

print("File updated successfully!")

time.sleep(1)

bash_script = "run_ros_launch.sh"
os.system(f"bash {bash_script}")
