import pandas as pd
import numpy as np
import random
import math
import sys
sys.path.append('/home/ubuntu/PycharmProjects/DeepRLAgent/')
import utils.load as load
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.layers import Lambda, Dense, Flatten
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn import linear_model
import datetime


seed = 7
np.random.seed(seed)


def load_data(path):
    """

    :param path:
    :return:
    """
    print(path)
    data_df = pd.read_csv(path, nrows=3600)
    data_df['origin'] = data_df['origin'].apply(str)

    data_df['observation'] = list(zip(data_df['origin'], data_df['time']))

    data_df['action'] = list(zip(data_df['destination'], data_df['mode']))

    starttime = datetime.datetime.now()
    le = preprocessing.LabelEncoder()
    data_df['label'] = le.fit_transform(data_df['action'])
    endtime = datetime.datetime.now()
    print('LabelEncoder time:', str(endtime-starttime))

    ref = pd.Series(data_df['action'], data_df['label']).to_dict()
    pd.DataFrame(list(ref.items())).to_csv('action_ref.csv')

    X = data_df['observation'].apply(observation_feature).values
    y = np_utils.to_categorical(data_df['label'].values)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


def get_business():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/OFFICECOUNTPOP/OFFICECOUNTPOP.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = (float(tokens[1]), float(tokens[2]))

    return mesh_info


def landuse_feature():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/LandUse/Landuse.csv') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(',')
            mesh_info[tokens[0]] = (float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]),
                                    float(tokens[5]), float(tokens[6]), float(tokens[7]), float(tokens[8]),
                                    float(tokens[9]), float(tokens[10]), float(tokens[11]), float(tokens[12]))

    return mesh_info


def entertainment():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/Entertainment/Entertainment.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = float(tokens[1])

    return mesh_info


def evacuate():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/Evacuate/Evacuate.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = float(tokens[1])

    return mesh_info


def road_feature():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/Road/roadlength.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = (float(tokens[1]), float(tokens[2]), float(tokens[3]))

    return mesh_info


def pop_feature():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/MESHPOP/MeshPop.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = float(tokens[2])

    return mesh_info


def passanger_feature():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/STOPINFO/dailyrailpassanger.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")

def passanger_feature():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/STOPINFO/dailyrailpassanger.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = float(tokens[1])

    return mesh_info


def school_feature():
    mesh_info = dict()
    with open('/home/ubuntu/Data/Tokyo/SCHOOL/SchoolCount.csv', 'r') as f:
        for line in f:
            line = line.strip('\n')
            tokens = line.split(",")
            mesh_info[tokens[0]] = float(tokens[1])

    return mesh_info


def initial_features():
    feature_dict = dict()

    feature_dict['office'] = get_business()
    feature_dict['school'] = school_feature()
    feature_dict['pop'] = pop_feature()
    feature_dict['landuse'] = landuse_feature()
    feature_dict['entertainment'] = entertainment()
    feature_dict['evacuate'] = evacuate()

    return feature_dict


def observation_feature(observation):
    """
    :param observation:
    :return:
    """
    f = np.zeros(20)
    feature_dict = initial_features()

    destination = observation[0]
    time = observation[1]

    try:
        # business features
        f[0] = float(feature_dict['office'][destination][1])
        f[1] = float(feature_dict['office'][destination][0])
        # school features
        f[2] = feature_dict['school'][destination]
        # population feature
        f[3] = feature_dict['pop'][destination]
        # landuse features
        f[4] = float(feature_dict['landuse'][destination][0])
        f[5] = float(feature_dict['landuse'][destination][1])
        f[6] = float(feature_dict['landuse'][destination][2])
        f[7] = float(feature_dict['landuse'][destination][3])
        f[8] = float(feature_dict['landuse'][destination][4])
        f[9] = float(feature_dict['landuse'][destination][5])
        f[10] = float(feature_dict['landuse'][destination][6])
        f[11] = float(feature_dict['landuse'][destination][7])
        f[12] = float(feature_dict['landuse'][destination][8])
        f[13] = float(feature_dict['landuse'][destination][9])
        f[14] = float(feature_dict['landuse'][destination][10])
        f[15] = float(feature_dict['landuse'][destination][11])
        # syukyaku facilities
        f[16] = feature_dict['entertainment'][destination]
        # evacuate
        f[17] = feature_dict['evacuate'][destination]
        # time of day feature
        f[18] = math.sin(math.pi*time/24.)
        f[19] = math.cos(math.pi*time/24.)
    except KeyError:
        return f
    return pd.Series(f)


def build_model(out_size):
    """

    :return:
    """

    model = Sequential()

    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))

    model.add(Dense(out_size, activation='softmax'))

    return model


def build_linear_model(X_train, y_train):
    """

    :param X_train:
    :param y_train:
    :return:
    """
    clf = linear_model.SGDClassifier(max_iter=1000)
    clf.fit(X_train, y_train)

    return clf


def policy(model, ref, g):
    """

    :param model:
    :param ref:
    :param g:
    :return:
    """
    time_state_action_prob = dict()
    for t in range(12, 48):
        state_action_prob = dict()
        for node in g.get_nodes():
            action_prob = dict()
            prob = model.predict(observation_feature((node, t)).values.reshape(1, 20))
            for i in range(len(prob)):
                action = ref[i]
                action_prob[action] = prob[i]
            state_action_prob[node] = action_prob
        time_state_action_prob[t] = state_action_prob

    return time_state_action_prob


def predict():
    # X_train, X_test, y_train, y_test = load_data("/home/ubuntu/Data/baseline_test.csv")
    model = load_model('Behavior_Cloning.h5', compile=False)
    ref = pd.DataFrame.from_csv('action_ref.csv')
    ref_dict = pd.Series(ref['1'], ref['0']).to_dict()

    current_place = '53393571'
    for t in range(36):
        prob = model.predict(observation_feature((current_place, t+12)).values.reshape(1, 20))
        print(prob)
        action = ref_dict[np.argmax(prob)]
        print(t+12, action)
        current_place = str(action[1:9])
        print(action[1:9])

    return action


def generate_initial(trajectories):

    mesh_count = dict()

    for trajectory in trajectories:
        if 0 in trajectory:
            origin = trajectory[0][0]
            if origin not in mesh_count:
                mesh_count[origin] = 1
            else:
                mesh_count[origin] += 1

    return mesh_count


def simulation(bc_policy, ref, start, count, target):
    current_state = start
    prev_action = None

    ref = pd.DataFrame.from_csv('action_ref.csv')
    ref_dict = pd.Series(ref['1'], ref['0']).to_dict()

    for i in range(count):
        out = open('/home/ubuntu/Result/BehaviorCloning/' + target + '/BC_' + start + "_" + str(i) + '.csv', 'w')
        for t in range(0, 47):
            action = ref_dict[random_temporal_weight(bc_policy[t][current_state])]
            out.write(start+str(i) + "," + str(t) + "," + current_state + ","
                      + str(action[1:9]) + "," + str(action[12:16]) + "\n")
            current_state = str(action[1:9])
            prev_action = action

        out.close()
    

def random_temporal_weight(weight_data):
    _total = sum(weight_data.values())
    _random = random.uniform(0, _total)
    _curr_sum = 0
    _ret = None

    for _k in weight_data.keys():
        _curr_sum += weight_data[_k]
        if _random <= _curr_sum:
            _ret = _k
            break
    return _ret


def main():
    # ---------------------------------------------------------------------------------------------------
    print("start")
    X_train, X_test, y_train, y_test = load_data("/home/ubuntu/Data/baseline_test.csv")
    id_traj = load.load_trajectory("/home/ubuntu/Data/baseline_test.csv")
    trajectories = id_traj.values()
    g = load.load_graph_traj(trajectories)
    print (X_train.shape, y_train.shape)
    model = build_linear_model(X_train, y_train)
    ref = pd.DataFrame.from_csv('action_ref.csv')

    bc_policy = policy(model, ref, g)
    
    mesh_count = generate_initial(trajectories)

    for mesh in mesh_count:
        simulation(bc_policy, ref, mesh, mesh_count[mesh])

    # model = build_model(y_train.shape[1])
    # model.compile(loss="categorical_crossentropy", optimizer='adam', matrix=['accuracy'])
    #
    # model.fit(X_train, y_train, epochs=100
    #           , batch_size=20, verbose=1)
    #
