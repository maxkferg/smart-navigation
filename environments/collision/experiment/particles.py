import json
import time
import redis
from .config import *
from .car import control_car
from ..simulation.utils import normalizeAngle
from ..simulation.particles import Particle
from ..simulation.targets import Target


db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)



class RealObject(Particle):
    """
    A real object we are tracking in the environment
    Follows the particle API from simulation
    """
    mass = 1 # For API compatability
    elasticity = 1
    control_count = 0

    def __init__(self,size,color,name="car"):
        self.x = 0
        self.y = 0
        self.angle = 0
        self.speed = 0
        self.size = size
        self.radius = int(size/2)
        self.color = color
        self.drag = 0
        self.name = name


    def update(self):
        """
        Update position according to database
        """
        data = db.get("position").decode('utf-8')
        position = json.loads(data)
        self.x = CAMERA_SCALE_X * position["x"]
        self.y = CAMERA_SCALE_Y * position["y"]
        self.speed = position.get("speed",0)
        self.angle = normalizeAngle(position["angle"])


    def save(self):
        """
        Save information to the database
        """
        position = json.dumps({"x":self.x, "y":self.y, "angle":self.angle})
        db.set("position",position)


    def control(self, steering, throttle):
        if self.control_count % 5 ==0:
            input("Press enter to continue")
        # Now execute the control
        steering = -steering # Real steering is reversed to simulation

        if throttle > 0 and throttle < 0.2:
            throttle += 0.2

        if throttle > 0 and throttle < 0.3:
            throttle += 0.1

        if throttle < 0 and throttle > -0.2:
            throttle -= 0.1

        control_car(rotation=steering, throttle=throttle)
        time.sleep(0.1)
        control_car(rotation=steering, throttle=0, reset=True)
        self.control_count += 1



class RealTarget(Target):
    """
    A real target we are pulling from the database
    """

    def __init__(self,size,color,name="target"):
        self.x = 0
        self.y = 0
        self.name = name
        self.size = size
        self.color = color
        self.radius = int(size/2)


    def update(self):
        """
        Update position according to database
        """
        result = db.get("target").decode('utf-8')
        if result is None:
            print("Unable to load target position")
        else:
            position = json.loads(result)
            self.x = CAMERA_SCALE_X * position["x"]
            self.y = CAMERA_SCALE_Y * position["y"]


    def save(self):
        """
        Save information to the database
        """
        x = self.x / CAMERA_SCALE_X
        y = self.y / CAMERA_SCALE_Y
        position = json.dumps({"x":x, "y":y})
        db.set("target",position)

