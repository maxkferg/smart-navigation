import json
import redis
import numpy as np
import math, random
import matplotlib
from gql import gql, Client
from .utils import addVectors, pol2cart
from .config import *
from .graphql import control_car

STEERING_SENSITIVITY = 0.02

db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def combine(p1, p2):
    if math.hypot(p1.x - p2.x, p1.y - p2.y) < p1.size + p2.size:
        total_mass = p1.mass + p2.mass
        p1.x = (p1.x*p1.mass + p2.x*p2.mass)/total_mass
        p1.y = (p1.y*p1.mass + p2.y*p2.mass)/total_mass
        (p1.angle, p1.speed) = addVectors((p1.angle, p1.speed*p1.mass/total_mass), (p2.angle, p2.speed*p2.mass/total_mass))
        p1.speed *= (p1.elasticity*p2.elasticity)
        p1.mass += p2.mass
        p1.collide_with = p2



class Particle:
    """ A circular object with a velocity, size and mass """

    def __init__(self, position, size, target=None, mass=1, elasticity=0.8, speed=0, backend='simulation', name="default"):
        (x, y) = position
        self.x = x
        self.y = y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 0
        self.angle = 0
        self.speed = speed
        self.mass = mass
        self.elasticity = elasticity
        self.target = target
        self.noise = np.array([0,0])
        self.backend = backend
        self.name = name

    def move(self):
        """ Update position based on speed, angle """
        if self.backend=="simulation":
            self.x += math.sin(self.angle) * self.speed
            self.y -= math.cos(self.angle) * self.speed
        else:
            position = json.loads(db.get("position"))
            self.x = CAMERA_SCALE_X * (position["x"]+position["width"]/2)
            self.y = CAMERA_SCALE_Y * (position["y"]+position["height"]/2)
            #print("Loaded redis position", self.x, self.y)


    def atTarget(self, threshold=10):
        """Return True if this particle is close enough to its target"""
        dx = abs(self.x - self.target.x)
        dy = abs(self.y - self.target.y)
        return (dx**2 + dy**2 < threshold**2)


    def get_speed_vector(self, scale=1):
        """Return the speed vector in cartesion coordinates"""
        dx, dy = pol2cart(self.angle, scale*self.speed)
        return dx, dy


    def get_control_vector(self, scale=1):
        """Return the control vector relative to the speed vector"""
        desired_angle = self.angle + self.control_signal[0]
        desired_speed = scale * (self.control_signal[1] + self.speed)
        dx, dy = pol2cart(desired_angle, desired_speed)
        return dx, dy


    def get_state_vector(self,w,h):
        """Return a normalized version of this objects state vector"""
        return (
            self.x / w,
            self.y / h,
            self.target.x / w,
            self.target.y / h
        )

    def experienceDrag(self):
        self.speed *= self.drag


    def control(self, steering, throttle):
        """ Change angle and speed by a given vector """
        self.angle += steering * self.speed * STEERING_SENSITIVITY
        self.speed += throttle
        self.control_signal = (steering,throttle)

        if self.backend=="redis":
            control_car(rotation=steering, throttle=throttle)

    def brownian(self):
        """Add some correlated acceleration"""
        pass
        #change = np.random.uniform(-1,1,size=2)
        #vector = 0.5*self.noise + 1.0*change + 0.6*self.speed
        #vector = pol2cart(self.angle, 0.003*self.speed) # Antidrag
        #polar = cart2pol(vector[0],vector[1])
        #self.accelerate(polar)
        #self.noise = vector / np.linalg.norm(vector)


    def attract(self, other):
        """" Change velocity based on gravatational attraction between two particle"""

        dx = (self.x - other.x)
        dy = (self.y - other.y)
        dist  = math.hypot(dx, dy)

        if dist < self.size + other.size:
            return True

        theta = math.atan2(dy, dx)
        force = 0.1 * self.mass * other.mass / dist**2
        self.accelerate((theta - 0.5 * math.pi, force/self.mass))
        other.accelerate((theta + 0.5 * math.pi, force/other.mass))

