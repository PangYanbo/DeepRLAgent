import nltk
import random

from irl.mdp import load
import sys
import random
sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def makePairs(trajectories):
    pairs = []
    for trajectory in trajectories:
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            destination = trajectory[slot][1].get_destination()
            mode = trajectory[slot][1].get_mode()
            temp = (origin, destination)
            pairs.append(temp)
    return pairs


def generate(cfd, path, word='53393574', num=36):
    out = open(path, 'w')
    for i in range(num):
        # make an array with the words shown by proper count
        arr = []
        for j in cfd[word]:
            for k in range(cfd[word][j]):
                arr.append(j)

        # choose the word randomly from the conditional distribution
        word = arr[int((len(arr)) * random.random())]
        out.write(str(12+i)+','+word+'\n')
    out.close()

path_observed = "D:/ClosePFLOW/53393574/training/"
id_traj = load.load_directory_trajectory(path_observed)

for i in range(36):
    path = 'D:/ClosePFLOW/53393574/markovchain/' + str(i) + '.csv'
    trajectories = random.sample(id_traj.values(), 50)
    pairs = makePairs(trajectories)
    cfd = nltk.ConditionalFreqDist(pairs)
    generate(cfd, path)

