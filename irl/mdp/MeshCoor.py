from irl.mdp import tools


class MeshToCoor(object):

    def __init__(self, bottomleft, upperright):
        self.min_lon, self.min_lat = tools.parse_MeshCode(bottomleft)
        self.max_lon, self.max_lat = tools.parse_MeshCode(upperright)
        self.x_unit = (self.max_lon-self.min_lon)/80.0
        self.y_unit = (self.max_lat - self.min_lat) / 120.0
        print self.x_unit, self.y_unit

    def get_minx(self):
        return self.min_lon

    def get_miny(self):
        return self.min_lat

    def get_maxx(self):
        return self.max_lon

    def get_maxy(self):
        return self.max_lat

    def get_coor(self, mesh):
        x, y = tools.parse_MeshCode(mesh)
        return int((x-self.min_lon)/self.x_unit), int((y-self.min_lat)/self.y_unit)
