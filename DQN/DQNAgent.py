import numpy as np
from itertools import chain
from collections import deque
import tensorflow as tf

np.random.seed(1)
# tf.set_random_seed(1)


class DQNAgent:
    """
    To implement your own agent, you have to implement the following methods:
    - `forward`
    - `backward`
    - `compile`
    - `load_weights`
    - `save_weights`
    - `layers`
    observation(node, t)
    2018/09/25 update
    expand state from current location to last four time step
    use agent_hist as state container
    2018/09/26/ update
    """
    def __init__(self, env, n_features, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=300,
                 memory_size=500, batch_size=32, agent_hist_len=4, e_greedy_increment=None, output_graph=True):
        self.env = env
        self.n_actions = len(env.action_space)
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.agent_hist_len = agent_hist_len

        # total learning step
        self.learn_step_counter = 0

        # update memory shape at 2018/09/25
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features*2*self.agent_hist_len + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _init_memory(self):
        print("Initializing replay memory: ", end='')
        self.memory = deque(maxlen=self.agent_hist_len)
        while True:
            state = self.env.reset()
            agent_hist = deque(maxlen=self.agent_hist_len)
            agent_hist.append(state)
            while True:
                action = self.choose_action(agent_hist=None)
                new_state, reward, done, _ = self.env.step(action)
                if len(agent_hist) == self.agent_hist_len:
                    self.store_transition(agent_hist, action, reward, new_state, done)
                if len(self.memory) == self.replay_start_size:
                    print('done')
                    return
                if done:
                    break
                state = new_state
                agent_hist.append(state)

    def flatten_features(self, features):
        return np.array(list(chain(*features)))

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features * self.agent_hist_len], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features * self.agent_hist_len], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            e2 = tf.layers.dense(e1, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e2')
            self.q_eval = tf.layers.dense(e2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            t2 = tf.layers.dense(t1, 200, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t2')
            self.q_next = tf.layers.dense(t2, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q_next')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None, )
            print("q_target_prev", q_target)
            self.q_target = tf.stop_gradient(q_target)
            print("q_target", self.q_target.shape)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack([s.flatten(), [a, r], s_.flatten()])

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, agent_hist):

        features = self.env.agent_hist_feature(agent_hist)

        # if np.random.uniform() < self.epsilon:
        #     # forward feed the observation and get q value for every actions
        #     actions_value = self.sess.run(self.q_eval, feed_dict={self.s: features})
        #     action = np.argmax(actions_value)
        # else:
        #     action = np.random.randint(0, self.n_actions)

        action_space = self.env.graph.get_node(str(agent_hist[3][1])).get_edges()

        action_num = []

        n_inf = float("-inf")

        for action in action_space:
            action_num.append(self.env.action_space_dict[action])

        if np.random.uniform() < self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: features})
            _action_value = np.full((1, self.env.n_actions), n_inf)

            for action in action_num:
                _action_value[0, action] = actions_value.item(action)

            action = np.argmax(_action_value)
        else:
            action = np.random.choice(action_num)

        # print(observation, self.env.action_space[action])
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.batch_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features*self.agent_hist_len],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features*self.agent_hist_len:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

