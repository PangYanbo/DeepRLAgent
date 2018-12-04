import Q_Learning.Env as Env
from Q_Learning.brain import choose_action, q_learning


env = Env.Env(1, "/home/ubuntu/Data/all_train_irl.csv")

env.load_alpha()
reward = env.reward()

Q = q_learning(Env, reward, 0.001, 0.9, 0.8, 1000)
