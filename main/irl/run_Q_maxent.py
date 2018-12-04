import main.irl.Q_maxent as irl_maxent
import Q_Learning.Env as Env
from utils import load, tools, writeout

import os
import random


def main(discount, epochs, learning_rate, path):
    # inverse reinforcement learning based on Q-learning
    env = Env.Env(discount, path + 'all_train_irl.csv')
    irl_maxent.irl(env, epochs, discount, learning_rate, path + 'PT_Result/Q_maxent/')


if __name__ == "__main__":
    main(0.9, 500, 3, "/home/ubuntu/Data/")
