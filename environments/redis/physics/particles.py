import json
import redis
import numpy as np
import math, random
import matplotlib
from gql import gql, Client
from .utils import addVectors, pol2cart
from .config import *
from .graphql import control_car

MAX_SPEED = 1.4 # Maximum simulation speed
STEERING_SENSITIVITY = 0.3 # Radians I rotate at speed=1 and steering=1
ACCELERATION_SENSITIVITY = 0.5 # The amount I speed up at full throttle
PIXELS_PER_SPEED = 10 # The pixels travelled at speed = 1
ADVERSARY_SPEED = 1.2

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
        self.collisions = 0
        self.control_signal = (0,0)

    def move(self):
        """ Update position based on speed, angle """
        if self.backend=="simulation":
            self.x += math.sin(self.angle) * self.speed * PIXELS_PER_SPEED
            self.y -= math.cos(self.angle) * self.speed * PIXELS_PER_SPEED
        else:
            position = json.loads(db.get("position"))
            self.x = CAMERA_SCALE_X * (position["x"]+position["width"]/2)
            self.y = CAMERA_SCALE_Y * (position["y"]+position["height"]/2)
            #print("Loaded redis position", self.x, self.y)

    def move_adversary(self):
        """Move randomly in a correlated manner"""
        if self.name!="primary":
            self.speed = ADVERSARY_SPEED + 0.1 * random.random()
            self.angle += STEERING_SENSITIVITY * random.uniform(-1,1)
            self.speed = np.clip(self.speed, 0, MAX_SPEED)
            self.angle = np.clip(self.angle, 0, 2*math.pi)


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
            self.angle / (2 * math.pi),
            self.target.x / w,
            self.target.y / h
        )

    def experienceDrag(self):
        self.speed *= self.drag


    def control(self, steering, throttle):
        """ Change angle and speed by a given vector """
        self.angle += steering * self.speed * STEERING_SENSITIVITY
        self.speed += throttle * ACCELERATION_SENSITIVITY
        self.control_signal = (steering,throttle)

        if abs(self.speed) > MAX_SPEED:
            self.speed = math.copysign(MAX_SPEED, self.speed)

        if self.backend=="redis":
            control_car(rotation=steering, throttle=throttle)
