class Trip(object):
    """

    """
    _origin = ""
    _destination = ""
    _mode = ""
    #_record = []

    def __init__(self, origin, destination, mode):
        self._origin = origin
        self._destination = destination
        self._mode = mode

    def get_origin(self):
        return self._origin

    def get_destination(self):
        return self._destination

    def get_mode(self):
        return self._mode

