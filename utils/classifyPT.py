"""
classify PT data by user attribute
file_one: commuter
file_two: student
file_three: housewife
file_five: others
"""
import os


def classify_occupation(directory):
    files = os.listdir(directory)

    f1 = open("D:/training data/PT_commuter.csv", "w")
    f2 = open("D:/training data/PT_student.csv", "w")
    f3 = open("D:/training data/PT_housewife.csv", "w")
    f4 = open("D:/training data/PT_others.csv", "w")

    for filename in files:
        path = directory + filename
        if not os.path.isdir(path):
            print path
            with open(path, "r") as f:
                for line in f.readlines():
                    tokens = line.split(",")
                    if tokens[9] == "1" or tokens[9] == "2" or tokens[9] == "3" or tokens[9] == "4" or tokens[9] == "5" or tokens[9] == "6" or tokens[9] == "7" or tokens[9] == "8" or tokens[9] == "9" or tokens[9] == "10":
                        f1.write(line)
                    elif tokens[9] == "11" or tokens[9] == "12" or tokens[9] == "13":
                        f2.write(line)
                    elif tokens[9] == "14":
                        f3.write(line)
                    else:
                        f4.write(line)

    f1.close()
    f2.close()
    f3.close()
    f4.close()

classify_occupation("D:\\PFLOW\\PFLOW\\0000\\")