import numpy as np

class Domain:
    def __init__(self, x1, x2, y1, y2, nx1, nx2, ny1, ny2):
        '''
        Initialize Domain object.
        '''
        self.x_coordinates = np.array(0)
        self.y_coordinates = np.array(0)

        top_left = np.tensordot( np.arange(0.0, x1 + dx1, step = dx1), np.ones(ny1), axes = 0 )
        top_right = 0
        bottom_left = 0
        bottom_right = 0





