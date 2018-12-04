import main.Env.urbanEnv as Env
import main.irl.maxent as maxent
import os
import datetime
import random


def main():
    env = Env.UrabanEnv(0.9, "/home/ubuntu/Data/all_train_irl.csv")
    maxent.irl(env, 1000, 0.01, "/home/ubuntu/Data/")


if __name__ == '__main__':
    main()