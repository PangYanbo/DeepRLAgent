import os
import os.path


def Coordinate2MeshCode(dLat, dLng ):
    dLat = float(dLat)
    dLng = float(dLng)

    iMeshCode_1stMesh_Part_p = dLat *60 // 40;
    iMeshCode_1stMesh_Part_u = ( dLng - 100 ) // 1;
    iMeshCode_2ndMesh_Part_q = dLat *60 % 40 // 5;
    iMeshCode_2ndMesh_Part_v = ( ( dLng - 100 ) % 1 ) * 60 // 7.5;
    iMeshCode_3rdMesh_Part_r = dLat *60 % 40 % 5 * 60 // 30;
    iMeshCode_3rdMesh_Part_w = ( ( dLng - 100 ) % 1 ) * 60 % 7.5 * 60 // 45;
    iMeshCode = iMeshCode_1stMesh_Part_p * 1000000 + iMeshCode_1stMesh_Part_u * 10000 + iMeshCode_2ndMesh_Part_q * 1000 + iMeshCode_2ndMesh_Part_v * 100 + iMeshCode_3rdMesh_Part_r * 10 + iMeshCode_3rdMesh_Part_w;
    print(str(int(iMeshCode)))
    return str(int(iMeshCode));


# with open("D:\\PFLOW\\all_mesh.csv", "w") as wr:
with open("/home/ubuntu/Data/pflow_data/p_all_mesh.csv", "w") as wr:
    # directory = "D:\\PFLOW\\PFLOW\\"
    directory = "/home/ubuntu/Data/pflow_data/pflow-csv/"
    # files = os.listdir(directory)

    # for root, dirs, files in os.walk("D:\\PFLOW\\PFLOW\\"):
    for root, dirs, files in os.walk("/home/ubuntu/Data/pflow_data/pflow-csv/"):
        for name in files:
            path = os.path.join(root, name)
            print(path)
            if not os.path.isdir(path):
                mode_list = []
                mode = "97"
                with open(path) as f:
                    prev_row = f.readline().split(",")
                    row = f.readline().split(",")
                    prev_purpose = prev_row[10]
                    for line in f.readlines():
                        line = line.strip("\n")
                        tokens = line.split(",")

                        if tokens[10] != prev_purpose:
                            if "12" in mode_list or "11" in mode_list:
                                mode = "12"
                            elif "2" in mode_list or "3" in mode_list or "4" in mode_list or "5" in mode_list or "6" in mode_list or "7" in mode_list or "8" in mode_list or "9" in mode_list or "10" in mode_list:
                                mode = "6"
                            elif "1" in mode_list:
                                mode = "1"
                            else:
                                mode = "97"
                            print(set(mode_list))
                            # wr.write(tokens[0]+","+prev_row[3]+","+row[3]+","+prev_row[4]+","+prev_row[5]+","+row[4]+","+row[5]+","+mode)
                            purpose = ""
                            if row[10] == "1" or row[10] == "2":
                                purpose = "1"
                            elif row[10] == "3":
                                purpose = "3"
                            else:
                                purpose = "4"
                            wr.write(tokens[0]+","+prev_row[3]+","+row[3]+","+Coordinate2MeshCode(prev_row[5],prev_row[4])+","+Coordinate2MeshCode(row[5],row[4])+","+mode+","+purpose)
                            wr.write("\n")
                            # clean mode_list
                            mode_list = []
                            prev_row = tokens
                        mode_list.append(tokens[13])
                        prev_purpose = tokens[10]
                        row = tokens
