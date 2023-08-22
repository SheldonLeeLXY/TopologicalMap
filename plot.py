import matplotlib.pyplot as plt

path = "/home/sheldon/camvideo_"

data_1 = [0.572178889662402, 0.5742238321709429, 0.5533497291502713, 0.7183674196121662, 0.531329123001601, 0.4466503665590392, 0.4635996426161758, 0.5530101466065269, 0.6781439761269962, 0.7279083831481661, 0.6544585967129509]
data_2 = [0.47241082910380844, 0.5712573645677772, 0.6490604057671938, 0.7016072279974054, 0.5328324836595539, 0.46723038252505183, 0.7394216187695456, 0.576830958836959, 0.7102067364424433, 0.7024322076067537, 0.6456177413959368]
data_3 = [0.5790510308051882, 0.5750916894386503, 0.6385312171844332, 0.7025856698795362, 0.5000728625455436, 0.48815342940790796, 0.7462066654675188, 0.6437376444290083, 0.6888818038044293, 0.6953832447750704, 0.6421949248376954]

def get_rotation_error(sum_imageset):
	with open(path + str(sum_imageset) + "/rotation_error.txt", "r") as f:
		rotation_error = f.read()
	data = rotation_error.split("\n")[:-1]
	float_data = []
	for x in data:
		float_data.append(float(x))
	return float_data

def rotation_error_plot():
	error_1 = get_rotation_error(9)
	error_2 = get_rotation_error(18)
	error_3 = get_rotation_error(36)
	x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	l1 = plt.plot(x, error_1, "r--", label="9")
	l2 = plt.plot(x, error_2, "g--", label="18")
	l3 = plt.plot(x, error_3, "b--", label="36")
	plt.plot(x,error_1,"ro-",x,error_2,"g+-",x,error_3,"b^-")
	plt.title("Different number of images")
	plt.xlabel("ImageSets")
	plt.ylabel("Rotation Error")
	plt.legend()
	plt.show()

# boxplot
# plt.figure(figsize=(10, 5))
# plt.title("Similarity Result of Boxplot", fontsize=20)
# labels = "9", "18", "36"
# plt.boxplot([data_1, data_2, data_3],labels = labels)
# plt.show()

# line chart
# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# l1 = plt.plot(x, data_1, "r--", label="9")
# l2 = plt.plot(x, data_2, "g--", label="18")
# l3 = plt.plot(x, data_3, "b--", label="36")
# plt.plot(x,data_1,"ro-",x,data_2,"g+-",x,data_3,"b^-")
# plt.title("Different number of images")
# plt.xlabel("ImageSets")
# plt.ylabel("Similarity")
# plt.legend()
# plt.show()

if __name__ == '__main__':
	rotation_error_plot()