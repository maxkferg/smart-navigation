import time
import json
import redis
import numpy as np
import math, random
import matplotlib
from gql import gql, Client
from .utils import addVectors, pol2cart
from .config import *
from .graphql import control_car
from .utils import normalizeAngle

MAX_SPEED = 1.4 # Maximum simulation speed
STEERING_SENSITIVITY = 0.25 # Radians I rotate at speed=1 and steering=1
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

    def __init__(self, position, size, target=None, mass=1, elasticity=0.8, speed=0, backend='simulation', name="default", ghost=False):
        (x, y) = position
        self.x = x
        self.y = y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 0
        self.angle = 0
        self.speed = speed
        self.mass = mass
        self.ghost = ghost
        self.elasticity = elasticity
        self.target = target
        self.noise = np.array([0,0])
        self.backend = backend
        self.name = name
        self.collisions = 0
        self.control_signal = (0,0)
        self.steering_sensitivity = STEERING_SENSITIVITY
        self.acceleration_sensitivity = ACCELERATION_SENSITIVITY


    def clone(self):
        """Return a deep copy of this object. The clone uses the same target as before"""
        position = (self.x, self.y)
        copy = Particle(position, size=self.size)
        copy.size = self.size
        copy.colour = self.colour
        copy.thickness = self.thickness
        copy.angle = self.angle
        copy.speed = self.speed
        copy.mass = self.mass
        copy.ghost = self.ghost
        copy.elasticity = self.elasticity
        copy.target = self.target
        copy.noise = self.noise
        copy.backend = self.backend
        copy.name = self.name
        copy.drag = self.drag
        copy.collisions = self.collisions
        copy.control_signal = self.control_signal
        return copy


    def reset(self):
        """Reset the particul dynamics"""
        self.steering_sensitivity = max(np.random.normal(loc=STEERING_SENSITIVITY, scale=0.2*STEERING_SENSITIVITY), 0.1)
        self.acceleration_sensitivity = max(np.random.normal(loc=ACCELERATION_SENSITIVITY, scale=0.5*ACCELERATION_SENSITIVITY), 0.1)


    def move(self):
        """ Update position based on speed, angle """
        if self.backend!="redis":
            self.x += math.sin(self.angle) * self.speed * PIXELS_PER_SPEED
            self.y -= math.cos(self.angle) * self.speed * PIXELS_PER_SPEED
        else:
            position = json.loads(db.get("position"))
            self.x = CAMERA_SCALE_X * position["x"]
            self.y = CAMERA_SCALE_Y * position["y"]
            #print("Rendered ANGLE (1):",self.angle)
            #print("DB ANGLE:",position["angle"])
            self.speed = 0
            self.angle = normalizeAngle(position["angle"])
            #print("Rendered ANGLE (2):",self.angle)
            #print("Loaded redis position", self.x, self.y)

    def move_adversary(self):
        """Move randomly in a correlated manner"""
        if self.name not in ["primary","ghost"]:
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


    def get_direction_vector(self, scale=1):
        """Return the speed vector in cartesion coordinates"""
        if self.name=="primary":
            print("AN:", self.angle)
        dx, dy = pol2cart(self.angle, scale*self.size)
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
        self.angle += steering * self.speed * self.steering_sensitivity
        self.speed += throttle * self.acceleration_sensitivity
        self.control_signal = (steering,throttle)

        if abs(self.speed) > MAX_SPEED:
            self.speed = math.copysign(MAX_SPEED, self.speed)

        if self.backend=="redis":
            steering = -steering
            choice = input("Execute steering: %.3f throttle %.3f?"%(steering,throttle))
            if choice == "y":
                control_car(rotation=steering, throttle=throttle)
                time.sleep(0.1)
                control_car(rotation=steering, throttle=0, reset=True)







