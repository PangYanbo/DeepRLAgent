import matplotlib.pyplot as plt

path = ''
path_little = ''

id_volume = {}
little_id_volume = {}

with open(path,'r') as f:
    next(f)
    for line in f.readlines():
        try:
            line = line.strip('\n')
            tokens = line.split(",")
            link_id = tokens[0]
            volume_6 = int(tokens[1])
            volume_7 = int(tokens[2])
            volume_8 = int(tokens[3])
            volume_9 = int(tokens[4])
            volume_10 = int(tokens[5])
            volume_11 = int(tokens[6])
            volume_12 = int(tokens[7])
            volume_13 = int(tokens[8])
            volume_14 = int(tokens[9])
            volume_15 = int(tokens[10])
            volume_16 = int(tokens[11])
            volume_17 = int(tokens[12])
            volume_18 = int(tokens[13])
            volume_19 = int(tokens[14])
            volume_20 = int(tokens[15])
            volume_21 = int(tokens[16])
            volume_22 = int(tokens[17])
            volume_23 = int(tokens[18])
            volume_total = int(tokens[19])

            id_volume[link_id] = volume_total

        except(AttributeError, ValueError, IndexError, TypeError):
            print "errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........"
f.close()

with open(path_little) as f:
    next(f)
    for line in f.readlines():
        try:
            line = line.strip('\n')
            tokens = line.split(",")
            link_id = tokens[0]
            volume_6 = int(tokens[1])
            volume_7 = int(tokens[2])
            volume_8 = int(tokens[3])
            volume_9 = int(tokens[4])
            volume_10 = int(tokens[5])
            volume_11 = int(tokens[6])
            volume_12 = int(tokens[7])
            volume_13 = int(tokens[8])
            volume_14 = int(tokens[9])
            volume_15 = int(tokens[10])
            volume_16 = int(tokens[11])
            volume_17 = int(tokens[12])
            volume_18 = int(tokens[13])
            volume_19 = int(tokens[14])
            volume_20 = int(tokens[15])
            volume_21 = int(tokens[16])
            volume_22 = int(tokens[17])
            volume_23 = int(tokens[18])
            volume_total = int(tokens[19])

            little_id_volume[link_id] = volume_total * 10

        except(AttributeError, ValueError, IndexError, TypeError):
            print "errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........"
f.close()

volume_list = []
little_volume_list = []

for link_id in id_volume:
    if link_id in little_id_volume:
        little_volume_list.append(little_id_volume[link_id])
        volume_list.append(id_volume[link_id])
    else:
        little_volume_list.append(0)
        volume_list.append(id_volume[link_id])

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_title('Scatter Plot')

plt.xlabel('X')

plt.ylabel('Y')

ax1.scatter(volume_list, little_volume_list, c='r',marker = 'o')

plt.legend('x1')

plt.show()