import numpy as np
import math, random
import matplotlib



class Target:
    """ A circular object with a size"""

    def __init__(self, position, size, color, name="target"):
        (x, y) = position
        self.x = x
        self.y = y
        self.radius = int(size/2)
        self.color = color
        self.name = name

    def clone(self):
        position = (self.x, self.y)
        copy = Target(position, self.size, self.color)
        return copy

    def update(self):
        """Update current position"""
        pass

    def respawn(self, screen_width, screen_height):
        """ Respawn the target """
        self.x = np.random.randint(low=self.radius, high=(screen_width-self.radius))
        self.y = np.random.randint(low=self.radius, high=(screen_height-self.radius))