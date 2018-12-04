import os.path
import shutil


def Coordinate2MeshCode(dLat, dLng):
    dLat = float(dLat)
    dLng = float(dLng)

    iMeshCode_1stMesh_Part_p = dLat * 60 // 40
    iMeshCode_1stMesh_Part_u = (dLng - 100) // 1
    iMeshCode_2ndMesh_Part_q = dLat * 60 % 40 // 5
    iMeshCode_2ndMesh_Part_v = ((dLng - 100) % 1) * 60 // 7.5
    iMeshCode_3rdMesh_Part_r = dLat * 60 % 40 % 5 * 60 // 30
    iMeshCode_3rdMesh_Part_w = ((dLng - 100) % 1) * 60 % 7.5 * 60 // 45
    iMeshCode = iMeshCode_1stMesh_Part_p * 1000000 + iMeshCode_1stMesh_Part_u * 10000 + iMeshCode_2ndMesh_Part_q * 1000 + iMeshCode_2ndMesh_Part_v * 100 + iMeshCode_3rdMesh_Part_r * 10 + iMeshCode_3rdMesh_Part_w;

    return str(int(iMeshCode))


def convert_career(_raw):
    career = ""

    if _raw in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
        career = "commuter"
    elif _raw in ["11", "12", "13"]:
        career = "student"
    elif _raw in ["14"]:
        career = "housewife"
    else:
        career = "other"

    return career

directory = "/home/ubuntu/Data/pflow_data/pflow-csv/"

mesh_type_count = {}


count = 0
for root, dirs, files in os.walk("/home/ubuntu/Data/pflow_data/pflow-csv/"):
    for name in files:
        path = os.path.join(root, name)

        if not os.path.isdir(path):
            with open(path) as f:
                count += 1
                tokens = f.readline().split(",")
                occupation = convert_career(tokens[9])
                mesh = Coordinate2MeshCode(tokens[5], tokens[4])

                if mesh not in mesh_type_count:
                    type_count = dict()
                    type_count[occupation] = 1
                    mesh_type_count[mesh] = type_count
                else:
                    if occupation not in mesh_type_count[mesh]:
                        type_count[occupation] = 1
                        mesh_type_count[mesh] = type_count
                    else:
                        type_count[occupation] += 1
                        mesh_type_count[mesh] = type_count
                if count % 10000 == 0:
                    print mesh_type_count
                # # move files based on initial mesh
                # new_path = "/home/ubuntu/Data/pflow_data/pflow-csv/" + mesh + "/"
                #
                # if not os.path.exists(new_path):
                #     os.mkdir(new_path)
                #
                # shutil.move(path, new_path)
                # print path

with open("/home/ubuntu/Data/pflow_data/init_distribution.csv", "w")as f:
    f.write("mesh, commuter, student, housewife, other")
    f.write("\n")
    for mesh in mesh_type_count.keys():
        if "commuter" in mesh_type_count[mesh].keys():
            commuters = mesh_type_count[mesh]["commuter"]
        else:
            commuters = 0

        if "student" in mesh_type_count[mesh].keys():
            students = mesh_type_count[mesh]["student"]
        else:
            students = 0

        if "housewife" in mesh_type_count[mesh].keys():
            housewives = mesh_type_count[mesh]["housewife"]
        else:
            housewives = 0

        if "other" in mesh_type_count[mesh].keys():
            others = mesh_type_count[mesh]["other"]
        else:
            others = 0
        f.write(mesh + "," + str(commuters) + "," + str(students) + "," + str(housewives) + "," + str(others))
        f.write("\n")


