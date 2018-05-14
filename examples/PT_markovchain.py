import nltk

from irl.mdp import load
import os
import sys
import random
sys.path.append("D:/Ubicomp/Inverse-Reinforcement-Learning-master")


def makepairs(trajectories):
    pairs = []
    for trajectory in trajectories:
        for slot in trajectory.keys():
            origin = trajectory[slot][1].get_origin()
            edge = trajectory[slot][1]
            temp = (origin, edge)
            pairs.append(temp)
    return pairs


def generate(cfd, path, word, num=36):
    out = open(path, 'w')
    for i in range(num):
        # make an array with the words shown by proper count

        arr = []

        for j in cfd[word]:
            print "1111111111111111111"
            print len(cfd[word])
            print "2222222222222222222"
            for k in range(cfd[word][j]):
                arr.append(j)

        # choose the word randomly from the conditional distribution

        print type(arr)

        word = arr[int((len(arr)) * random.random())]

        out.write(str(i)+','+str(12+i)+','+word.get_origin()+','+word.get_destination()+','+word.get_mode()+'\n')
        word = word.get_destination()
    out.close()


def main(target):
    path_observed = "D:/PT_Result/" + target + "/"
    id_traj = load.load_directory_trajectory(path_observed + "training/")

    if not os.path.exists(path_observed + "markovchain/"):
        os.mkdir(path_observed + "markovchain/")

    for i in range(36):
        path = "D:/PT_Result/" + target + "/" + '/markovchain/' + str(i) + '.csv'
        trajectories = random.sample(id_traj.values(), 50)

        initial = []

        for traj in trajectories:
            if 12 in traj.keys():
                initial.append(traj[12][1].get_origin())

        pairs = makepairs(trajectories)

        cfd = nltk.ConditionalFreqDist(pairs)

        generate(cfd, path, random.choice(initial))
        print "###################################"

if __name__ == '__main__':
    main("student")

