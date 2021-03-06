import time
import json
import redis
import numpy as np
import math, random
import matplotlib
from .utils import addVectors, pol2cart
from .utils import normalizeAngle

MAX_SPEED = 1.5 # Maximum simulation speed
STEERING_SENSITIVITY_MIN = 0.001
STEERING_SENSITIVITY_MAX = 0.003
ACCELERATION_SENSITIVITY_MIN = 0.1 # The amount I speed up at full throttle
ACCELERATION_SENSITIVITY_MAX = 0.9 # The amount I speed up at full throttle
RADIUS_MIN = 10 # Particle radius max - effects drag
RADIUS_MAX = 50 # Particle radius max - effects drag
PIXELS_PER_SPEED = 40 # The pixels travelled at speed = 1
ADVERSARY_SPEED = 0.2



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
    collisions = 0
    control_signal = (0,0)
    steering_sensitivity = STEERING_SENSITIVITY_MIN
    acceleration_sensitivity = ACCELERATION_SENSITIVITY_MIN

    def __init__(self, position, size, target=None, mass=1, elasticity=0.8, speed=0, backend='simulation', name="default", ghost=False, mass_of_air=0):
        (x, y) = position
        self.x = x
        self.y = y
        self.size = size
        self.radius = int(size/2)
        self.mass_of_air = mass_of_air
        self.color = (0, 0, 255)
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



    def clone(self):
        """Return a deep copy of this object. The clone uses the same target as before"""
        position = (self.x, self.y)
        copy = Particle(position, size=self.size)
        copy.size = self.size
        copy.color = self.color
        copy.mass_of_air = self.mass_of_air
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
        self.angle = math.pi/2
        self.steering_sensitivity = np.random.uniform(STEERING_SENSITIVITY_MIN, STEERING_SENSITIVITY_MAX)
        self.acceleration_sensitivity = np.random.uniform(ACCELERATION_SENSITIVITY_MIN, ACCELERATION_SENSITIVITY_MAX)
        self.radius = int(np.random.uniform(RADIUS_MIN, RADIUS_MAX))
        self.size = 2*self.radius # Drag is inversely proportional to size
        self.drag = (self.mass/(self.mass + self.mass_of_air)) ** self.size
        #print("Acceleration Sensitivity",self.acceleration_sensitivity)
        #print("Drag",self.drag)



    def move(self):
        """ Update position based on speed, angle """
        self.x += math.sin(self.angle) * self.speed * PIXELS_PER_SPEED
        self.y -= math.cos(self.angle) * self.speed * PIXELS_PER_SPEED


    def update(self):
        """Update current position"""
        pass


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
        dx, dy = pol2cart(self.angle, scale*self.radius)
        return dx, dy


    def get_control_vector(self, scale=1):
        """Return the control vector relative to the speed vector"""
        desired_angle = self.angle + math.copysign(self.control_signal[0],self.speed)
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




