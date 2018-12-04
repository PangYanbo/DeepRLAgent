import networkx as nx
import osmnx as ox


def write_out(dict, path):
    out = open(path, 'w')
    for time in dict:
        out.write()


def generate(path, G):

    time_node_mode = dict().fromkeys(range(12, 48), dict())

    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip("\r\n").split(',')

            uid = tokens[0]
            node = ox.get_nearest_node(G, (float(tokens[6]), float(tokens[5])))
            hour = int(tokens[3].split(' ')[1].split(':')[0])
            minute = int(tokens[3].split(' ')[1].split(':')[1])
            slot = hour * 2 + minute / 30
            mode = tokens[13]

            if slot in time_node_mode.keys():
                if node not in time_node_mode[slot]:
                    modes = []
                    modes.append(mode)
                    time_node_mode[slot][node] = modes
                else:
                    time_node_mode[slot][node].append(mode)

        return time_node_mode


def main():
    path = '/home/ubuntu/Data/pflow_data/pflow-csv/52386799/00358432.csv'
    G = ox.graph_from_place('Shibuya, Tokyo, Japan', network_type='drive')
    print(generate(path, G))


if __name__ == "__main__":
    main()
