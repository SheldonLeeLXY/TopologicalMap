from math import pi

path = "/home/sheldon/camvideo_9/"
rotation_error_list = []

with open(path + "result1.0.txt", "r") as f:
	result = f.read()
result_line = result.split("\n")[:-1]
for line in result_line:
	rotation_angle = line.split(",")[-3]
	actual_angle = line.split(",")[-1][:-1]
	if abs(float(actual_angle) - float(rotation_angle)) > pi:
		rotation_error = abs(float(actual_angle) - float(rotation_angle))
		while rotation_error > 3:
			rotation_error = rotation_error - pi
		rotation_error = abs(rotation_error)
	else:
		rotation_error = abs(float(actual_angle) - float(rotation_angle))
	print(rotation_error)
	rotation_error_list.append(rotation_error)

with open(path + "rotation_error.txt", "w") as f:
	for line in rotation_error_list:
		f.write(str(line)+"\n")