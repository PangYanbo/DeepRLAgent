from irl.mdp import tools, load
import sys
import os
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import scipy.stats as stats

def pearson(pop1,pop2):

    pearson = 0.0

    sum_pop1 = 0.0

    sum_pop2 = 0.0

    sum_pop1_square = 0.0

    sum_pop2_square = 0.0

    sum_pop1pop2 = 0.0

    for meshid in pop1.keys():
        if meshid in pop2:
            sum_pop1 += pop1[meshid]
            sum_pop2 += pop2[meshid]
            sum_pop1_square += pop1[meshid] * pop1[meshid]
            sum_pop2_square += pop2.get(meshid) * pop2.get(meshid)
            sum_pop1pop2 += pop1.get(meshid) * pop2.get(meshid)
    }else{
        sum_pop1 += pop1.get(meshid);
    sum_pop2 += 0;
    sum_pop1_square += pop1.get(meshid) * pop1.get(meshid);
    sum_pop2_square += 0;
    sum_pop1pop2 += 0;
    }
    }
def gps_pop(number1, number2):
    count = 0
    mesh_hour_pop = {}
    # "/home/t-iho/Result/training/trainingdata20170109.csv"
    with open("D:/training data/PTtraj3.csv") as f:
        for line in f.readlines()[number1:number2]:
            try:
                count += 1
                if count % 100000 == 0:
                    print "finish " + str(count) + " lines"
                line = line.strip('\n')
                tokens = line.split(",")
                hour = int(tokens[1])/2
                mesh_id = tokens[2]
                if mesh_id not in mesh_hour_pop.keys():
                    hour_pop = {}
                    hour_pop[hour] = 0.5
                    mesh_hour_pop[mesh_id] = hour_pop
                else:
                    if hour not in mesh_hour_pop[mesh_id].keys():
                        mesh_hour_pop[mesh_id][hour] = 0.5
                    else:
                        mesh_hour_pop[mesh_id][hour] += 0.5

            except(AttributeError, ValueError, IndexError, TypeError):
                print "errrrrrrrrrrrrrrrroooooooooooooooooooooooooooorrrrrrrrrrrrrrrrrrrrrrrrrrrr........"

    f.close()
    return  mesh_hour_pop


def traj_to_mesh(date, rate):
    id_traj = load.load_trajectory(date)
    trajectories = random.sample(id_traj.values(), int(rate*len(id_traj)))
    mesh_t_count = {}
    for traj in trajectories:
        print traj
        for t in range(12, 47):
            if t in traj.keys():
                if traj[t][0] not in mesh_t_count.keys():
                    t_count = {}
                    t_count[t] = 1
                    mesh_t_count[traj[t][0]] = t_count
                else:
                    if t not in mesh_t_count[traj[t][0]].keys():
                        t_count[t] = 1
                        mesh_t_count[traj[t][0]] = t_count
                    else:
                        mesh_t_count[traj[t][0]][t] += 1
    return mesh_t_count

def main():
    mesh_t_pop_1 = gps_pop(0, 10000)
    mesh_t_pop_2 = gps_pop(10000, 12000)
    print mesh_t_pop_1


if __name__ == '__main__':
    main()