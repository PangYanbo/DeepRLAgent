import numpy as np


def policy_eval(policy, env, reward, discount=1.0, threshold=1e-4):
    V = np.zeros(env.n_state)
    while True:
        delta = 0
        # TODO
        for s in range(env.n_state):
            v = 0
            for a in env.sub_action_space(s):
                s_ = env.indext_action(a).get_destination()
                v += policy[s][a] * (reward[s][a] + discount * V[s_])
            delta = max(delta, np.abs(v-V[s]))
        if delta < threshold:
            break

    return np.array(V)
