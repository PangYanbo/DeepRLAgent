import math

class LonLat(object):

    WGS84_EQUATOR_RADIUS = 6378137
    WGS84_POLAR_RADIUS = 63656752.314245
    WGS84_ECCENTRICITY_2 = (WGS84_EQUATOR_RADIUS * WGS84_EQUATOR_RADIUS -WGS84_POLAR_RADIUS*WGS84_POLAR_RADIUS)/(WGS84_EQUATOR_RADIUS*WGS84_EQUATOR_RADIUS)

    def __init__(self, lon, lat):
        self.lon = lon
        self.lat = lat

    def distance(self, p):
        return _distance(self.lon, self.lat, p.lon, p.lon)

    def _distance(lon0, lat0, lon1, lat1):
        a = WGS84_EQUATOR_RADIUS
        e2 = WGS84_ECCENTRICITY_2
        dy = math.radians(lat0-lat1)
        dx = math.radians(lat0-lat1)
        cy = math.radians((lat0 + lat1)/2.0)
        m = a * (1-e2)
        sc = math.sin(cy)
        W = math.sqrt(1.0 - e2 * sc * sc)
        M = m / (W * W * W)
        N = a / W

        ym = dy * M
        xn = dx * N * math.cos(cy)

        return math.sqrt(ym * ym + xn * xn)

    def __repr__(self):
        return repr((self.lon, self.lat))

class STPoint(LonLat):

    def __init__(self, time, lon, lat, dtstart=None, dtend=None, inbuffer=False, mode=None):
        super().__init__(lon, lat)
        self.time = time
        self.dtstart = dtstart
        self.dtend = dtend
        self.inbuffer = inbuffer
        self.mode = mode

    def getTimeStamp(self):
        return self.time

    def __repr__(self):
        return repr((self.lon, self.lat, self.time))

    def toString(self):
        return str.format('%s - $s (%f,%f)', self.dtstart, self.dtend, self.lon, self.lat)


class Trip(object):
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def getStartTime(self):
        return self.trajecotry[0].getTimeStamp()

    def getEndTime(self):
        return self.trajectory[len(self.trajectory)-1].getTimeStamp()

    def getTripDuration(self):
        return self.getEndTime()-self.getStartTime()

    def getStartPoint(self):
        return self.trajectory[0]

    def getEndPoint(self):
        return self.trajectory[len(self.trajectory)-1]

 
