#!/usr/bin/env python
# -*- coding: utf-8 -*-


def read_list(path):
    mesh_list = []
    with open(path, "r")as f:
        for line in f.readlines():
            tokens = line.strip("\r\n").split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list


def main():
    mesh_time_pop = {}
    count = 0

    mesh_list = read_list("/home/ubuntu/Data/Tokyo/MeshCode/Tokyo.csv")
    # 2014 pflowstat
    with open("/home/ubuntu/Data/zdc_tokyo_distance.tsv") as f:
        f.readline()
        for line in f.readlines()[0:2933980]:
            if count % 10000 == 0:
                print count

            tokens = line.strip("\r\n").split("\t")

            if tokens[3][0:8] in mesh_list:
                if tokens[0] == "1-3月" and tokens[1] == "平日" and tokens[4]=="自宅": # and tokens[5]!="50km以上":
                    hour = tokens[2].split("-")[0]
                    mesh_id = tokens[3][0:8]
                    pop = int(tokens[6]) if tokens[6] != "NA" else 0

                    if mesh_id not in mesh_time_pop.keys():
                        hour_pop = dict()
                        hour_pop[hour] = pop
                        mesh_time_pop[mesh_id] = hour_pop.copy()
                    else:
                        if hour not in mesh_time_pop[mesh_id]:
                            mesh_time_pop[mesh_id][hour] = pop
                        else:
                            temp = mesh_time_pop[mesh_id][hour] + pop
                            mesh_time_pop[mesh_id][hour] = temp

    # 2014 pflow
    # with open("/home/ubuntu/Data/zdc_tokyo.tsv") as f:
    #     f.readline()
    #     for line in f.readlines()[0:540000]:
    #         if count % 10000 == 0:
    #             print count
    #         tokens = line.strip("\r\n").split("\t")
    #
    #         if tokens[3][0:8] in mesh_list:
    #
    #             if tokens[0] == "1～3月" and tokens[1] == "平日":
    #                 hour = tokens[2].split("-")[0]
    #
    #                 mesh_id = tokens[3][0:8]
    #                 pop = int(tokens[4]) if tokens[4] != "NA" else 0
    #
    #                 if mesh_id not in mesh_time_pop.keys():
    #                     hour_pop = dict()
    #                     hour_pop[hour] = pop
    #                     mesh_time_pop[mesh_id] = hour_pop.copy()
    #                 else:
    #                     if hour not in mesh_time_pop[mesh_id]:
    #                         mesh_time_pop[mesh_id][hour] = pop
    #                         # print mesh_time_pop[mesh_id][hour]
    #                     else:
    #                         temp = mesh_time_pop[mesh_id][hour] + pop
    #                         mesh_time_pop[mesh_id][hour] = temp
    #                         # print temp

            count += 1
    print len(mesh_time_pop.keys())

    with open("/home/ubuntu/Data/zdc_tokyo_1km_stat.csv", "w") as w:

        for mesh in mesh_time_pop:
            for t in range(0, 24):
                if str(t) in mesh_time_pop[mesh]:
                    w.write(mesh+","+str(t)+","+str(mesh_time_pop[mesh][str(t)])+"\n")
                else:
                    w.write(mesh + "," + str(t) + "," + str(0) + "\n")


if __name__ == "__main__":
    main()