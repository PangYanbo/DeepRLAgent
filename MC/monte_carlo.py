"""
Find the value function via Monte Carlo Methods. Based on Sutton & Barto, 2016
Yanbo Pang, Nov, 19, 2018

"""
import sys
sys.path.append("/home/ubuntu/PycharmProjects/DeepRLAgent/")
import numpy as np
import random
from collections import defaultdict
import main.Env.urbanEnv as urbanEnv


def on_policy_first_visit_mc_control(env, n_episodes=5000):
    """
    """
    reward = env.get_reward()
    # print(reward)

    Q = {}
    for s in env.state_space:
        Q[s] = {}
	for a in env.sub_action_space(s):
	    Q[s][a] = random.random()

    returns_sum = defaultdict(float)
    states_count = defaultdict(float)

    policy = epsilon_greedy_policy(Q, env)

    for k in range(n_episodes):
	if k % 10 == 0:
	    print("processed " + str(k) + " episodes")
            # update policy every 10 iteration
	    policy = epsilon_greedy_policy(Q, env)

	for s in env.initial_state:
            episode = generate_episode(env, policy, s)
            for i, sa_pair in enumerate(episode):
		state, action, reward = episode[i]
           	G = sum([reward[i+j][env.action_index[sa[1]]] for j, sa in enumerate(episode[i:])])
 
		returns_sum[sa_pair] += G 
		states_count[sa_pair] += 1.0
		Q[s][a] = returns_sum[sa_pair] / states_count[sa_pair]
    return Q


#-----------------------------------policy---------------------------------------------------------

def epsilon_greedy_policy(Q, env, epsilon=0.2):
    """
    """
    policy = dict()

    for s in env.state_space:
        policy[s] = {}
	optimal_action = max(Q[s].items(), key = lambda x:x[1])[0]

	for a in env.sub_action_space(s):
	    if a == optimal_action:
		policy[s][a] = 1 - epsilon + epsilon / len(env.sub_action_space(s))
	    else:
	        policy[s][a] = epsilon / len(env.sub_action_space(s))

    return policy


def random_policy(env):
    """
    """
    policy = dict()

    for s in env.state_space:
	policy[s] = {}
	for a in env.sub_action_space(s):
	    policy[s][a] = 1.0 / len(env.sub_action_space(s))

    return policy


def greedy_policy(Q, env):
    """
    """
    policy = dict()

    for s in env.state_space:
	policy[s] = {}
	optimal_action = max(Q[s].items(), key = lambda x:x[1])[0]
	for a in env.sub_action_space(s):
	    policy[s][a] = 0
	policy[s][optimal_action] = 1.

    return policy

def optimal_action(s, Q):
    """
    """

    return max(Q[s].items(), key = lambda x: x[1])[0]


#--------------------------------Generate episode--------------------------------------------------

def generate_episode(env, policy, start_state):
    """
    """
    episode = []
    current_state = start_state
    
    for time in range(48):
        # print("--------test----------", current_state in policy.keys())
        
	if current_state not in policy.keys():
	    break

        action = np.random.choice(policy[current_state].keys(), 1, policy[current_state].values())[0]
	# print(action)

	next_state, reward, _ = env.step(current_state, action)

	episode.append((current_state, action, reward))
        
        current_state = next_state

    return episode


#-----------------------------------Off-policy Monte Carlo control--------------------------------------

def off_policy_mc_control(env, n_episodes):
    """
    """
    Q = dict()
    C = dict()

    for s in env.state_space:
	Q[s] = dict()
	C[s] = dict()
	for a in env.sub_action_space(s):
	    Q[s][a] = 0.0
	    C[s][a] = 0.0

    behavior_policy = random_policy(env)
    target_policy = greedy_policy(Q, env) 

    for k in range(n_episodes):
	if k % 10 == 0:
	    print("processed "+str(k)+" episodes")

        # generate episode using behavior policy
	initial_state = env.reset()
	episode = generate_episode(env, behavior_policy, initial_state)

	G = 0
	W = 1
 
	for t in range(len(episode))[::-1]:
	    state, action, reward = episode[t]
	    
	    G = G + reward
	    C[state][action] += W 
	    Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

	    # update target_policy 
	    # target_poicy = greedy_policy(Q, env)

	    if action != optimal_action(state, Q):
		# print("not optimal action", t)
		break
	    W = W * 1./behavior_policy[state][action]

    return Q, target_policy


def main():

    env = urbanEnv.UrbanEnv(100, "/home/ubuntu/Data/all_train_irl.csv")

    # Q = on_policy_first_visit_mc_control(env)

    Q, _ = off_policy_mc_control(env, 500)

    print(Q)


if __name__ == "__main__":
    import profile
    profile.run('main()')
