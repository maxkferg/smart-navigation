import copy
import numpy as np
import math, random
import matplotlib
from .targets import Target
from .particles import Particle
from .utils import addVectors, normalizeAngle




class Universe:
    """ Defines the boundary of a simulation and its properties """

    def __init__(self, size, discretization=5):
        (width, height) = size
        self.width = width
        self.height = height
        self.targets = []
        self.particles = []
        self.springs = []

        self.color = (255,255,255)
        self.mass_of_air = 0.2
        self.elasticity = 0.2

        self.particle_functions1 = []
        self.particle_functions2 = []
        self.function_dict = {
            'move': (1, lambda p: p.move()),
            'drag': (1, lambda p: p.experienceDrag()),
            'bounce': (1, lambda p: self.bounce(p)),
            'collide': (2, lambda p1, p2: self.collide(p1, p2)),
            'combine': (2, lambda p1, p2: combine(p1, p2)),
            'move_adversary': (1, lambda p: p.move_adversary()),
        }

        self.penalties = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.spawn = np.copy(self.penalties)
        self.discretization = self.spawn.shape[0]


    def addFunctions(self, function_list):
        for func in function_list:
            (n, f) = self.function_dict.get(func, (-1, None))
            if n == 1:
                self.particle_functions1.append(f)
            elif n == 2:
                self.particle_functions2.append(f)
            else:
                print("No such function: %s" % f)


    def addParticle(self, **kargs):
        """ Add n particles with properties given by keyword arguments """
        name = kargs.get('name', 'default')
        size = kargs.get('radius', random.randint(10, 20))
        mass = kargs.get('mass', random.randint(100, 10000))
        drag = kargs.get('drag', random.random())
        ghost = kargs.get('ghost', False)
        target = kargs.get('target', None)
        backend = kargs.get('backend', 'simulation')
        x = kargs.get('x', random.uniform(size, self.width - size))
        y = kargs.get('y', random.uniform(size, self.height - size))

        particle = Particle((x, y), size, target=target, mass=mass, name=name, backend=backend, ghost=ghost)
        particle.speed = kargs.get('speed', random.random())
        particle.angle = kargs.get('angle', random.uniform(0, math.pi*2))
        particle.color = kargs.get('color', (0,0,0))
        particle.elasticity = kargs.get('elasticity', self.elasticity)
        particle.drag = (particle.mass/(particle.mass + self.mass_of_air)) ** particle.size

        self.particles.append(particle)


    def addTarget(self, radius, color):
        """Add a target for the particles to reach"""
        target = Target((0,0), radius, color)
        target.respawn(self.width, self.height)
        self.targets.append(target)
        return target


    def _get_grid_cell_bounds(self, xi, yi):
        """Return the pixel bounds of one of the grid cells"""
        xmin = xi / self.discretization * self.width
        xmax = (xi+1) / self.discretization * self.width
        ymin = yi / self.discretization * self.height
        ymax = (yi+1) / self.discretization * self.height

        return (int(xmin), int(xmax), int(ymin), int(ymax))


    def reset(self):
        spawn_options = np.nonzero(self.spawn)
        spawn_indicies = list(range(len(spawn_options[0])))
        random.shuffle(spawn_indicies)

        for particle in self.particles:
            # Select a random square
            index = spawn_indicies.pop()
            xi = spawn_options[1][index] # Column
            yi = spawn_options[0][index] # Row
            xmin, xmax, ymin, ymax = self._get_grid_cell_bounds(xi, yi)

            particle.reset()
            particle.x = random.uniform(xmin+particle.size, xmax-particle.size)
            particle.y = random.uniform(ymin+particle.size, ymax-particle.size)
            particle.speed = 0.5 * random.random()
            particle.collisions = 0

        for target in self.targets:
            index = spawn_indicies.pop()
            xi = spawn_options[1][index] # Column
            yi = spawn_options[0][index] # Row
            xmin, xmax, ymin, ymax = self._get_grid_cell_bounds(xi, yi)

            target.x = random.uniform(xmin+target.radius, xmax-target.radius)
            target.y = random.uniform(ymin+target.radius, ymax-target.radius)


    def resetTarget(self, target):
        """Reset the position of a target. Choose a reasonble target position"""
        spawn_options = np.nonzero(self.spawn)
        spawn_indicies = list(range(len(spawn_options[0])))
        random.shuffle(spawn_indicies)

        index = spawn_indicies.pop()
        xi = spawn_options[1][index] # Column
        yi = spawn_options[0][index] # Row
        xmin, xmax, ymin, ymax = self._get_grid_cell_bounds(xi, yi)
        target.x = random.uniform(xmin+target.radius, xmax-target.radius)
        target.y = random.uniform(ymin+target.radius, ymax-target.radius)


    def update(self):
        """
        Moves particles and tests for collisions with the walls and each other
        Return the number of particle-particle collisions
        """
        for i, particle in enumerate(self.particles, 1):
            for f in self.particle_functions1:
                f(particle)
            for particle2 in self.particles[i:]:
                for f in self.particle_functions2:
                    f(particle, particle2)
            # Fix all the angles
            particle.angle = normalizeAngle(particle.angle)



    def bounce(self, particle):
        """ Tests whether a particle has hit the boundary of the environment """

        if particle.x > self.width - particle.radius:
            particle.x = 2*(self.width - particle.radius) - particle.x
            particle.angle = - particle.angle
            particle.speed *= self.elasticity
            particle.collisions += 1

        elif particle.x < particle.radius:
            particle.x = 2*particle.radius - particle.x
            particle.angle = - particle.angle
            particle.speed *= self.elasticity
            particle.collisions += 1

        if particle.y > self.height - particle.radius:
            particle.y = 2*(self.height - particle.radius) - particle.y
            particle.angle = math.pi - particle.angle
            particle.speed *= self.elasticity
            particle.collisions += 1

        elif particle.y < particle.radius:
            particle.y = 2*particle.radius - particle.y
            particle.angle = math.pi - particle.angle
            particle.speed *= self.elasticity
            particle.collisions += 1


    def collide(self, p1, p2):
        """ Tests whether two particles overlap
            If they do, make them bounce, i.e. update their angle, speed and position """

        dx = p1.x - p2.x
        dy = p1.y - p2.y

        dist = math.hypot(dx, dy)
        if dist < p1.radius + p2.radius:
            angle = math.atan2(dy, dx) + 0.5 * math.pi
            total_mass = p1.mass + p2.mass

            (p1.angle, p1.speed) = addVectors((p1.angle, p1.speed*(p1.mass-p2.mass)/total_mass), (angle, 2*p2.speed*p2.mass/total_mass))
            (p2.angle, p2.speed) = addVectors((p2.angle, p2.speed*(p2.mass-p1.mass)/total_mass), (angle+math.pi, 2*p1.speed*p1.mass/total_mass))
            elasticity = p1.elasticity * p2.elasticity
            p1.speed *= elasticity
            p2.speed *= elasticity

            overlap = 0.5*(p1.radius + p2.radius - dist+1)
            p1.x += math.sin(angle)*overlap
            p1.y -= math.cos(angle)*overlap
            p2.x -= math.sin(angle)*overlap
            p2.y += math.cos(angle)*overlap

            p1.collisions += 1
            p2.collisions += 1

    def isOnPenalty(self, particle):
        # Find discretized position
        xi = math.floor( particle.x / self.width * self.discretization)
        yi = math.floor( particle.y / self.height * self.discretization)

        try:
            return self.penalties[yi,xi]==0
        except IndexError:
            return False


    def findParticle(self, x, y):
        """ Returns any particle that occupies position x, y """

        for particle in self.particles:
            if math.hypot(particle.x - x, particle.y - y) <= particle.radius:
                return particle
        return None
