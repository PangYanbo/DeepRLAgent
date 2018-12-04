import numpy as np
import datetime
import sys
from collections import deque
sys.path.append("/home/ubuntu/PycharmProjects/DeepRLAgent/")
from utils import load
from main.Env import Env
from DQN import DQNAgent


def run(env, agent):
    step = 0
    starttime = datetime.datetime.now()
    episode_reward = []
    for episode in range(300000):

        observation = env.reset()
        cum_reward = 0

        agent_hist = deque(maxlen=agent.agent_hist_len)
        agent_hist.append(observation)
        print(agent.flatten_features(agent_hist))

        while True:

            if len(agent_hist) < agent.agent_hist_len:
                observation = agent_hist[0]
                for i in range(agent.agent_hist_len - len(agent_hist)):
                    agent_hist.appendleft(observation)
                action = agent.choose_action(agent_hist)
            else:
                action = agent.choose_action(agent_hist)

            observation_, reward, done, _ = env.step(observation[0], action)
            # print(observation, agent.env.action_space[action], reward)

            # print("choose action using: ", str(endtime-starttime))

            cum_reward += reward

            _agent_hist = agent_hist
            _agent_hist.append(observation_)

            agent.store_transition(env.agent_hist_feature(agent_hist),
                                   action, reward, env.agent_hist_feature(_agent_hist))

            if step > 350 and step % 35 == 0:

                agent.learn()

            observation = observation_

            agent_hist = _agent_hist

            if done:
                print("finish one day")
                episode_reward.append(cum_reward)
                break
            step += 1

    endtime = datetime.datetime.now()

    print(str(endtime-starttime))

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(episode_reward)), episode_reward)
    plt.ylabel('Reward of a Episode')
    plt.xlabel('training steps')
    plt.show()


def main():
    mesh_id = "53393574"
    id_traj = load.load_trajectory("/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh_id + "/train_irl.csv")
    trajectories = id_traj.values()

    alpha = load.load_param("/home/ubuntu/Data/PT_Result/dynamics_param_previous/" + "53393227_0_param.csv")

    g = load.load_graph_traj(trajectories)

    g.set_start(mesh_id)

    env = Env.Env(g, alpha)
    agent = DQNAgent.DQNAgent(env, env.n_features, learning_rate=0.1, reward_decay=0.95, e_greedy=0.9,
                              replace_target_iter=35,
                              memory_size=70, output_graph=True)
    run(env, agent)
    agent.plot_cost()


if __name__ == "__main__":
    # profile.run('main()')
    main()
