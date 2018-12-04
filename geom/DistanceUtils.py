import math


class DistanceUtils(object):

    WGS84_EQUATOR_RADIUS = 6378137
    WGS84_POLAR_RADIUS = 63656752.314245
    WGS84_ECCENTRICITY_2 = (WGS84_EQUATOR_RADIUS * WGS84_EQUATOR_RADIUS -WGS84_POLAR_RADIUS*WGS84_POLAR_RADIUS)/(WGS84_EQUATOR_RADIUS*WGS84_EQUATOR_RADIUS)

    def distance(lon0, lat0, lon1, lat1):
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


