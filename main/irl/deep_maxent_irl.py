import numpy as np
import tensorflow as tf
import main.Solver.value_iteration as value_iteration
import Q_Learning.Env as Env


class DeepIRL:

    def __init__(self, n_features, learning_rate, l2=10, name='deep_irl_fc'):

        self.n_features = n_features
        self.learning_rate = learning_rate
        self.name = name

        self.sess = tf.Session()
        self.input_s, self.reward, self.theta = self._build_network(self.name)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        self.grad_r = tf.placeholder(tf.float32, [None, None, 1])
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)

        self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

        self.grad_norms = tf.global_norm(self.grad_theta)
        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self, name):
        input_s = tf.placeholder(tf.float32, [None, None, self.n_features])

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        with tf.variable_scope(name):
            fc1 = tf.layers.dense(input_s, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer, name='fc1')
            fc2 = tf.layers.dense(fc1, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer, name='fc2')
            rewards = tf.layers.dense(fc2, 1, kernel_initializer=w_initializer, bias_initializer=b_initializer,
                                      name='reward')

        theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

        return input_s, rewards, theta

    def get_rewards(self, feature_matrix):
        rewards = self.sess.run(self.reward, feed_dict={self.input_s: feature_matrix})
        return rewards

    def get_theta(self):
        return self.sess.run(self.theta)

    def apply_grads(self, feature_matrix, grad_r):
        grad_r = np.reshape(grad_r, [-1, 1])
        feature_matrix = np.reshape(feature_matrix, [-1, self.n_features])
        _, grad_theta, l2_loss, grad_norms = self.sess.run(
            [self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
            feed_dict={self.grad_r: grad_r, self.n_features: feature_matrix})
        return grad_theta, l2_loss, grad_norms


def find_demo_svf(env):
    """

    :param env:
    :param trajectories:
    :return:
    """
    n_actions = env.n_action
    print(env.trajectories)
    p = np.zeros([48, n_actions])
    for trajectory in env.trajectories:
        for traj in trajectory:
            i = env.action_index[trajectory[traj][1]]
            p[traj, i] += 1
    p = p / len(env.trajectories)

    return p


def find_expected_svf(env, policy):
    """

    :param env
    :param policy:
    :return:
    """
    n_actions = env.n_action

    trajectory_length = len(env.trajectories[0])

    start_action_count = np.zeros(n_actions)

    for trajectory in env.trajectories:
        start_action_count[trajectory[0][1]] += 1
    p_start_action = start_action_count / len(env.trajectories)

    expected_svf = np.tile(p_start_action, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        # Last time step(t-1) action probability
        for i in range(n_actions):
            dest = env.index_action[i].get_destination()
            # j is the index of sub action space
            for j in env.sub_action_space(dest):
                expected_svf[j, t] += expected_svf[i, t-1] * policy[t-1, i]

    return expected_svf


def deep_maxent_irl(env, discount, learning_rate, n_iters):
    """

    :param env:
    :param discount
    :param learning_rate:
    :param n_iters:
    :return:
    """
    nn_r = DeepIRL(env.n_features, learning_rate)

    demo_svf = find_demo_svf(env)

    for iteration in range(n_iters):
        if iteration % 10 == 0:
            print('iteration: {}'.format(iteration))

        reward = nn_r.get_rewards(env.feature_matrix)

        _, policy = value_iteration.value_iteration(env, reward, discount)

        expected_svf = find_expected_svf(env, policy)

        grad_r = expected_svf - demo_svf

        print(grad_r)

        grad_theta, l2_loss, grad_norm = nn_r.apply_grads(env.feature_matrix, grad_r)

    reward = nn_r.get_rewards(env.feature_matrix)

    return reward


if __name__ == '__main__':
    env = Env.Env(0.9, "/home/ubuntu/Data/pflow_data/pflow-csv/all_day_irl.csv")
    print('Deep Maxent IRL starts training')

    reward = deep_maxent_irl(env, 0.99, 0.01, 100)

