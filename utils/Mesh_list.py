def read_list(path):
    mesh_list = []
    with open(path, "r")as f:
        for line in f.readlines():
            tokens = line.split(",")
            mesh = tokens[2]
            mesh_list.append(mesh)
    return mesh_list
