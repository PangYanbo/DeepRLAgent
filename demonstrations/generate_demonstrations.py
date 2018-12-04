def read_buffer_point(path):

    id_points = dict()
    
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip('\n').split(',')
            if len(tokens) > 4:
                uid = tokens[0]
                
