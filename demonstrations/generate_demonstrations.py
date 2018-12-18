import datetime
import sys
sys.path.append('/home/ubuntu/PycharmProjects/DeepRLAgent')
import fiona
from shapely import shape, mapping, Point, Polygon, MultiPolygon



def read_buffer_point(path):

    id_points = dict()
    
    with open(path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip('\n').split(',')
            if len(tokens) > 4:
                uid = tokens[0]
                lat = float(tokens[3])
                lon = float(tokens[2])
                inbuffer = bool(tokens[4])
                dt = datetime.datetime(tokens[1], '%Y-%m-%d %H:%M:%S')
                point = STPoint(lon, lat)
            
