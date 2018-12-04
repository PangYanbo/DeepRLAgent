def write_id_trajs(id_traj, path):
    """

    :param id_traj:
    :param path:
    :return:
    """
    with open(path, 'wb') as f:

        for uid in id_traj:
            for t in range(12, 47):
                if t in id_traj[uid]:
                    traj = id_traj[uid][t]
                    line = uid + ',' + str(t) + ',' + traj[1].get_origin() + ',' + traj[1].get_destination() + ',' +\
                           traj[1].get_mode() + '\n'
                    f.write(line.encode())

    f.close()


def write_trajs(trajs, path):
    """

    :param trajs: list() of id_traj values()
    :param path: out file path
    :return:
    """
    count = 0
    with open(path, 'wb') as f:

        for traj in trajs:

            for t in range(12, 47):
                if t in traj:
                    line = str(count) + ',' + str(t) + ',' + traj[t][1].get_origin() + ',' + traj[t][1].get_destination() + ',' + \
                           traj[t][1].get_mode() + '\n'
                    f.write(line.encode())
            count += 1

    f.close()
