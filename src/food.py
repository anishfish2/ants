import numpy as np

class food():
    def __init__(self, environment_size=10, location=None):
        if location is None:
            self.location = {
                'x': np.random.randint(-environment_size, environment_size),
                'y': np.random.randint(-environment_size, environment_size)
            }
        else:
            self.location = location
        self.picked_up = False