import matplotlib.pyplot as plt
import numpy as np

import os
import sys

date_list = ["20170107", "20170108", "20170109", "20170110", "20170111", "20170112"]


trip_count = {}.fromkeys(date_list, [])

# plt.boxplot(trip_count.values(), labels=trip_count.keys())
# plt.show()
#
# print trip_count.keys()
#
# # count synthetic profile trip rate
#
# for date in trip_count.keys():
#     prev_id = ''
#     count = 0
#     for i in range(100):
#         path = '/home/t-iho/Result/UbiResult/Synthetic'+str(date)+'/'+'Synthetic'+str(i)+'.csv'
#         with open(path, 'r') as f:
#             for line in f:
#                 line = line.strip('\n')
#                 tokens = line.split(",")
#
#                 uid = tokens[0]
#                 mode = tokens[4]
#
#                 if uid != prev_id and prev_id != '':
#                     trip_count[date].append(count)
#                     count = 0
#
#                 if mode != "stay":
#                     count += 1
#
#                 prev_id = uid


# count for openpflow trip count

openpt_trip_count = []

path = "D:/OpenPFLOW/OpenPFLOWslot.csv"

prev_id = ''
count = 0

with open(path, 'r') as f:
    for line in f:
        line = line.strip('\n')
        tokens = line.split(",")

        uid = tokens[0]

        mode = tokens[4]

        if uid != prev_id and prev_id != '':
            openpt_trip_count.append(count)
            count = 0

        if mode != "stay":
            count += 1

        prev_id = uid

trip_count["OpenPFLOW"] = openpt_trip_count

path2 = "D:/training data/PTtraj3.csv"

pt_trip_count = []

print prev_id
prev_id = ''
print prev_id
count = 0
with open(path2, 'r') as f:
    for line in f:
        line = line.strip('\n')
        tokens = line.split(",")

        uid = tokens[0]
        mode = tokens[4]

        if uid != prev_id and prev_id != '':
            pt_trip_count.append(count)
            count = 0

        if mode != "stay":
            count += 1

        prev_id = uid

trip_count["Closed PT"] = pt_trip_count

plt.boxplot(trip_count.values(), labels=trip_count.keys())
plt.show()
plt.savefig("/home/t-iho/trip_rate.jpg")