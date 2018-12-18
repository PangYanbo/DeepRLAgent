import datetime
import fiona
from shapely.geometry import shape, mapping, Polygon, Point, MultiPolygon
import sys
sys.path.append('/home/ubuntu/PycharmProjects/DeepRLAgent')
from geom.geom import STPoint


#--------------------------Extract Logs from Target Area-------------------
def read_point(path, east_lon, east_lat, west_lon, west_lat):

    id_points = dict()

    with open(path, 'r') as f:
        for line in f.readlines():
            try:
                tokens = line.strip('\n').split(',')
                if len(tokens) >=5 and tokens[4] != None:
                    uid = tokens[0]
                    lat = float(tokens[5])
                    lon = float(tokens[4])
                    if lat >= west_lat and lat <=east_lat and lon <=east_lon and lon >= west_lon:
                        dt = datetime.datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
                        point = STPoint(dt, lon, lat)
                        if uid not in id_points:
                            id_points[uid] = list()

                        id_points[uid].append(point)

            except(AttributeError, ValueError, IndexError, TypeError):
                print("errrrrrrrrrrrrrrrrooooooooooooooooooooorrrrrrrrrrrrrrrrrrrhhhhhhhhhhheeeeeeeeeerrrrrrrrrrrrreeeeeeeeeee")
	
        for uid in id_points:
            temp = sorted(id_points[uid], key=lambda point:point.time)
            id_points[uid] = temp

        return id_points


(read_point('/home/ubuntu/Data/pflow_data/pflow-csv/52392688/00569707.csv', 140,40,138,30))


def user_filter(id_points, shp_path, threhold=20):
    """
    only check the first point and last point(home location)
    """
    inst = fiona.open(shp_path)
    multi = inst.next()
    # print("shapefile read by fiona", multi)

    for uid in id_points:
        if len(id_points[uid]) >= threhold:
            lat = id_points[uid][0].lat
            lon = id_points[uid][0].lon
            point = Point(lon, lat)

            lat_2 = id_points[uid][len(id_points[uid])-1].lat
            lon_2 = id_points[uid][len(id_points[uid])-1].lon
            point_2 = Point(lon_2, lat_2)

           #  print(multi['geometry'])

            if point.within(shape(multi['geometry'])) and point_2.within(shape(multi['geometry'])):
                slot = set()
                for p in id_points[uid]:
                    time = p.time.seconds
                    print(type(time))
                    slot.add(time/1800)
                if len(slot) >= 10:
                    continue
                else:
                    id_points.pop(uid)

    return id_points        
               

#print(user_filter(read_point('/home/ubuntu/Data/pflow_data/pflow-csv/52392688/00569707.csv', 140,40,138,30), '/home/ubuntu/Data/Tokyo/TokyoZone/TokyoZone/'))



def write_out(path, rail_buffer_path, id_points):
    """
        each point is added with status in rail_buffer when write out
    """
    rail_buffer = fiona.open(rail_buffer_path)
    rail_buffer = rail_buffer.next()

    with open(path, 'w') as f:
        for uid in id_points:
            for p in id_points[uid]:
                point = Point(p.lon, p.lat)
                f.write(uid + ',' + p.time.strftime('%Y-%m-%d %H:%M:%S') + ',' + str(p.lon) + ',' + str(p.lat) + ','  + str(point.within(shape(rail_buffer['geometry']))))
                f.write('\n')
     
    f.close()

write_out('/home/ubuntu/Result/test.csv', '/home/ubuntu/Data/Tokyo/RailBuffer/', user_filter(read_point('/home/ubuntu/Data/pflow_data/pflow-csv/52392688/00569707.csv', 140,40,138,30), '/home/ubuntu/Data/Tokyo/TokyoZone/TokyoZone/'))
#-----------------------------------------------------------------------
