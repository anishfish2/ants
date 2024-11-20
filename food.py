import numpy as np

class food():
    def __init__(self):
        self.location = {'x': np.random.randint(-size, size), 'y': np.random.randint(-size, size)}
        self.picked_up = False

    def __init__(self, location):
        self.location = location
        self.picked_up = False