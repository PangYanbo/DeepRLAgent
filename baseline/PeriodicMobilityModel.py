import numpy as np
import utils.load as load
import utils.tools as tools


# -------------------------------Periodic Mobility Model--------------------------------------------------
def PeriodicMobilityModel(init_mesh, trajectory):
    """
    proposed by Cho et al.(2011). Friendship Mobiliy: User Movement in Location-Based Social Networks. KDD
    1. define home/work location
     initial location as home
     most frequented place as work
    2. classify
     classify each place as home/work
     other as work
    3. calculate mu(home location) and phi from samples x[] and y[]
    :param trajectories:
    :return: periodic location distribution mix spatial and temporal distribution
    """
    # initial state
    init_x, init_y = tools.parse_MeshCode(init_mesh)

    # Temporal component of PMM
    home_t = []
    work_t = []
    # Spatial component of PMM
    work_x = []
    work_y = []

    home_x = []
    home_y = []
    # mu, initial location as home
    home_mesh = trajectory[0][0]
    home_mu_x, home_mu_y = tools.parse_MeshCode(home_mesh)

    mesh_count = dict()

    for time in trajectory:
        mesh = trajectory[time][0]
        _x, _y = tools.parse_MeshCode(mesh)
        if mesh != home_mesh:
            work_t.append(time)
            if mesh not in mesh_count:
                mesh_count[mesh] = 1
            else:
                mesh_count[mesh] += 1
            work_x.append(_x)
            work_y.append(_y)

        else:
            home_t.append(time)
            home_x.append(_x)
            home_y.append(_y)

    if len(mesh_count) > 0:
        work_mesh = max(mesh_count, key=mesh_count.get)
    else:
        work_mesh = home_mesh

    work_x_center, work_y_center = tools.parse_MeshCode(work_mesh)
    sim_work_x = init_x + work_x_center - home_mu_x
    sim_work_y = init_y + work_y_center - home_mu_y

    work_X = np.stack((np.array(work_x), np.array(work_y)), axis=0)
    work_cov = np.cov(work_X)

    home_X = np.stack((home_x, home_y), axis=0)
    home_cov = np.cov(home_X)

    # Temporal Distribution
    P_cH = len(home_t) / 48
    P_cW = len(work_t) / 48

    mean_home_t = np.mean(home_t)
    mean_work_t = np.mean(work_t)

    # Spatial Distribution
    work_loc_distribution = np.random.multivariate_normal((sim_work_x, sim_work_y), work_cov, 1)
    # home_loc_distribution = np.random.multivariate_normal((home_mu_x, home_mu_y), home_cov, 1)

    # Finally, define distribution as a dict{'t', mesh}
    t_mesh = {}
    for t in range(0, 48):
        std_home_t = np.std(home_t)
        std_work_t = np.std(work_t)

        NH_distribution_t = P_cH * 1/(std_home_t * np.sqrt(2 * np.pi)) * \
                                    np.exp(- (t - mean_home_t)**2 / (2 * std_home_t**2))
        NW_distribution_t = P_cW * 1/(std_work_t * np.sqrt(2 * np.pi)) * \
                                    np.exp(- (t - mean_work_t)**2 / (2 * std_work_t**2))

        P_H_t = NH_distribution_t / (NH_distribution_t + NW_distribution_t)
        P_W_t = NW_distribution_t / (NH_distribution_t + NW_distribution_t)

        # define 0 as home, 1 as work
        type = np.random.choice([0, 1], 1, p=[P_H_t/(P_H_t+P_W_t), P_W_t/(P_H_t+P_W_t)])

        if type[0] == 0:
            mesh_t = init_mesh
        else:
            x, y = work_loc_distribution[0]
            print(x, y)
            mesh_t = tools.Coordinate2MeshCode(x, y)

        # print("t, type: ", t, type, mesh_t, init_mesh)

        t_mesh[t] = mesh_t

    return t_mesh


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


def pmm_simulation(target, mesh_count, train_traj):
    """
    :param target:
    :param mesh_count:
    :param train_traj: list of trajectories used for training
    :return:
    """

    for mesh in mesh_count:
        for i in mesh_count[mesh]:
            out = open('/home/t-iho/Result/sim/PMM/' + target + '/PMM_' + mesh + "_" + str(i) + '.csv', 'w')
            t_mesh = PeriodicMobilityModel(mesh, np.random.choice(train_traj))
            for t in t_mesh:
                out.write(mesh + str(i)+","+str(t) + "," + t_mesh[t] + ',' + t_mesh[t] + ',' + 'stay' + '\n')
            out.close()


def main(target):
    id_traj = load.load_trajectory("/home/t-iho/Result/trainingdata/Fukuoka20170706.csv")

    train_traj = list(id_traj.values())

    mesh_count = generate_initial(train_traj)

    pmm_simulation(target, mesh_count, train_traj)


if __name__ == '__main__':
    main('Fukuoka')
