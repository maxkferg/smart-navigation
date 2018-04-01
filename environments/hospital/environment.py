import sys
import math
import time
import scipy
import random
import pygame
import pygame.gfxdraw
import matplotlib
import numpy as np
import gym.spaces
from collections import deque
from pygame import pixelcopy
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn import preprocessing
from skimage.draw import circle
from .simulation.utils import cart2pol
from .simulation.universe import Universe
from .experiment.particles import RealObject, RealTarget
from ..util.spaces import ObservationSpace, ActionSpace
import seaborn as sns; sns.set()


class Spec:
    id = "collision-environment"
    map_file = "environments/hospital/maps/room.png"
    particle_size = 56
    timestep_limit = 2000


def clip(val, minimum, maximum):
    """Clip a value to [min,max]"""
    return max(min(val,maximum),minimum)


def clipv(vector, space):
    """Clip @vector to the gym space"""
    return np.clip(vector, space.low, space.high)


def get_color(i):
    """Return a color from the pallete"""
    if i >= 1:
        i = 2
    colors = sns.color_palette("muted")
    return tuple(255*c for c in colors[i])


class Snapshot():
    """A snapshot of a previous environment"""
    def __init__(self, universe):
        self.particles = [p.clone() for p in universe.particles]
        self.targets = [t.clone() for t in universe.targets]
        # Link particle clones to target clones
        for particle,target in zip(self.particles, self.targets):
            particle.target = target

    def modifyUniverse(self, universe):
        """Modify the universe object to look more like this snapshot"""
        # Replace with the new particles
        universe.particles = self.particles
        universe.targets = self.targets



class Environment:
    """
    Environment that an algorithm can play with
    """
    reward_range = [-2,1]
    action_dimensions = 2
    state_dimensions = 6 # The number of dimensions per particle (x,y,theta,v,tx,ty)
    max_steps = 2000
    current_actor = 0

    spec = Spec()
    screen = None
    particle_speed = 20
    metadata = {
        'render.modes':['human']
    }


    def __init__(self, primary, num_particles, disable_render):
        """
        @history: A vector where each row is a previous state
        """

        self.current_step = 0
        self.reward_so_far = 0
        self.num_particles = num_particles
        self.state_size = num_particles*self.state_dimensions + self.action_dimensions
        self.observation_space = ObservationSpace(-1, 1, shape=(self.state_size,), dtype=np.float32)
        self.action_space = ActionSpace(-1, 1, shape=[self.action_dimensions], dtype=np.float32)
        self.previous_action = np.zeros(self.action_space.shape)

        self.universe = Universe(scipy.ndimage.imread(self.spec.map_file))
        self.universe.addFunctions(['move', 'bounce', 'collide', 'drag', 'move_adversary'])
        self.universe.mass_of_air = 0.1

        self.screen_width = self.universe.width
        self.screen_height = self.universe.height

        # Add all the other particles
        for i in range(self.num_particles):
            name = "default"
            speed = self.particle_speed
            color = get_color(i)
            target = self.universe.addTarget(radius=self.spec.particle_size, color=color)
            particle = self.universe.addParticle(radius=self.spec.particle_size, mass=100, speed=speed, elasticity=0.5, color=color, target=target, name=name)

        # Fix all the particles in one spot
        #self.fix_particles()

        # Reset the environment to correctly spawn the particles
        self.reset()
        if not disable_render:
            print('Initializing pygame screen with w=%i and h=%i'%(self.screen_width, self.screen_height))
            self.screen_buffer = pygame.Surface((self.screen_width, self.screen_height))
            self.screen_buffer.set_alpha(50)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Hospital Collision Avoidance')


    def step(self, action):
        """
        Step the environment forward
        Return (observation, reward, done, info)
        """
        # Fix all the particles in one spot
        #self.fix_particles()

        #for primary,action in zip(self.universe.particles, actions):
        primary = self.universe.particles[0]
        primary.name = "primary"

        # We clip the value even if the learning algorithm chooses not to
        # This should be seen as a last resort, to prevent simulation instability
        action = clipv(action, self.action_space)

        # Particle 1 is being controlled
        steering = action[0]
        throttle = action[1]
        primary.previous_action = action
        primary.control(steering, throttle)

        # Update our primary positions
        primary.update()
        primary.target.update()

        # Update the universe now all the actors have registred their moves
        self.universe.update()
        self.current_step += 1

        #for primary,action in zip(self.universe.particles, actions):
        # Calculate the current state
        state = self.get_current_state(primary)

        if primary.atTarget(threshold=40) and primary.speed<0.5:
            self.universe.resetTarget(primary.target)
            reward = 1
            done = False
        elif primary.collisions > 0:
            reward = -1
            done = True
        elif self.universe.is_on_occupied(primary):
            reward = -1
            done = True
        else:
            reward = 0
            done = self.current_step >= self.max_steps

        # Enforce speed limits
        if abs(primary.speed) > 1:
            reward -= 0.01

        if any(np.absolute(action) > 0.8):
            reward -= 0.01

        # Enforce penalty regions
        if self.universe.is_on_restricted(primary):
            reward -= 0.1

        info = {'step': self.current_step}
        return (state, reward, done, info)


    def reset(self, catastrophy=False):
        """Respawn the particles and the targets"""
        self.universe.reset()

        self.state_buffer = []

        self.current_step = 0

        self.reward_so_far = 0

        p = self.universe.particles[0]

        return self.get_current_state(p)


    def render(self, mode=None, close=None, background=None):
        """
        Render the environment
        """
        # Clear the screen
        if background is not None:
            pixelcopy.array_to_surface(self.screen, background)
        else:
            pixelcopy.array_to_surface(self.screen, self.universe.map)

        # Draw particles
        for p in self.universe.particles:
            edge = np.maximum(p.color, (255,255,255))
            self.draw_circle(int(p.x), int(p.y), p.radius, p.color, edgeColor=edge, filled=True)

        # Draw primary target
        for t in self.universe.targets[:1]:
            color = t.color
            color = (255,255,255)
            self.draw_circle(int(t.x), int(t.y), t.radius, color, filled=False)
            self.draw_circle(int(t.x), int(t.y), int(t.radius/4), color, filled=True)

        # Draw the primary particle orientation
        for p in self.universe.particles:
            dx, dy = p.get_direction_vector(scale=1)
            pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (0,0,0))

        # Draw the primary particle speed
        for p in self.universe.particles:
            dx, dy = p.get_speed_vector(scale=30)
            pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (204,204,0))

        # Draw the control vector
        try:
            p = self.universe.particles[0]
            dx, dy = p.get_control_vector(scale=30)
            pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (255,0,0))
        except AttributeError as e:
            print(e)

        self.flip_screen()
        time.sleep(0.01)


    def fix_particles(self):
        """Fix all the particles in one place"""
        self.universe.particles[0].x = 357
        self.universe.particles[0].y = 117
        self.universe.particles[0].angle = 3

        self.universe.particles[2].x = 135
        self.universe.particles[2].y = 276
        self.universe.particles[2].angle = 1.5*math.pi/2

        self.universe.particles[1].x = 565
        self.universe.particles[1].y = 271
        self.universe.particles[1].angle = 4



    def flip_screen(self):
        """
        Flip the pygame screen and catch events
        """
        # Push to the screen
        pygame.display.flip()

        # Make sure we catch quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()


    def draw(self,scale=10):
        """
        Draw the current state on a black and white image
        """
        width = int(self.screen_width/scale)
        height = int(self.screen_height/scale)
        img = np.zeros((height,width),dtype=np.uint8)
        for p in self.universe.particles:
            rr, cc = circle(p.x/scale, p.y/scale, p.size/scale)
            rr[rr>79] = 79; rr[rr<0] = 0
            cc[cc>79] = 79; cc[cc<0] = 0
            img[rr, cc] = 1
        return img


    def draw_circle(self, x, y, r, color, edgeColor=None, filled=False):
        """Draw circle on the screen"""
        edgeColor = color if edgeColor is None else edgeColor
        if filled:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), r, color)
        pygame.gfxdraw.aacircle(self.screen, int(x), int(y), r, edgeColor)



    def get_current_state(self,primary):
        """
        Return a representation of the simulation state, from the perspective of @primary
        The return array is size (self.state_history, self.state_size)
        """
        otherParticles = set(self.universe.particles)
        otherParticles.remove(primary)

        # Get the current state vector for this particle
        state = list(primary.get_state_vector(self.screen_width, self.screen_height))

        for particle in otherParticles:
            state.extend(particle.get_state_vector(self.screen_width, self.screen_height))

        state.extend(primary.previous_action)

        return np.array(state)



    def close(self):
        """Clean up the env"""
        pass



class LearningEnvironment(Environment):
    """
    A fully simulated environment for training control algorithms
    - The primary is simulated and randomly spawned
    - The target is simulated and randomly spawned
    @locus: The RL training computer
    """
    def __init__(self, num_particles=2, *args, **kwargs):
        super().__init__(None, num_particles, *args, **kwargs)



class ExecuteEnvironment(Environment):
    """
    A real environment used to control the car
    - The primary position is pulled from the database
    - The primary control signal is sent to the car
    - The target position is pulled from the database
    @locus: The Real Car
    """
    def __init__(self, num_particles=2, *args, **kwargs):
        primary = RealObject(particle_size, color=get_color(0), name="Real Car")
        primary.target = RealTarget(particle_size, color=get_color(0), name="Real Car Target")
        super().__init__(primary, num_particles, particle_size, *args, **kwargs)

    def reset(self):
        """
        Reset the environment, and immediately pull the new position
        Never sends a control signal to the real objects
        """
        self.primary.update()
        self.primary.target.update()
        return self.get_current_state()



class ViewingEnvironment(Environment):
    """
    Environment where all data is pulled from real environment
    - The primary object is the real car
    - The primary target is pulled from the database
    """
    def __init__(self, num_particles=2, *args, **kwargs):
        primary = RealObject(particle_size, color=get_color(0), name="Real Car")
        primary.target = RealTarget(particle_size, color=get_color(0), name="Real Car Target")
        super().__init__(primary, num_particles, particle_size, *args, **kwargs)

    def step(self,action):
        """
        Pull the updated positions from the database
        Never sends a control signal to the real objects
        """
        self.primary.update()
        self.primary.target.update()

    def reset(self):
        """
        Pull the updated positions from the database
        Never sends a control signal to the real objects
        """
        self.primary.update()
        self.primary.target.update()



class HumanLearningEnvironment(LearningEnvironment):
    """
    Overide base class to allow human control
    """

    def reset(self,*args,**kwargs):
        """
        Print the environment dynamics on reset
        """
        state = super().reset(*args, **kwargs)
        #print("Steering sensitivity",self.primary.steering_sensitivity)
        #print("Acceleration sensitivity",self.primary.acceleration_sensitivity)
        return state


    def control_loop(self):
        """
        Return a user selected action
        """
        action = None

        while action is None:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = [0, 0.5]
                if event.key == pygame.K_d:
                    action = [0.9, 0.5]
                if event.key == pygame.K_s:
                    action = [0, -0.5]
                if event.key == pygame.K_a:
                    action = [-0.9, 0.5]
        return np.array(action)


if __name__=="__main__":
    # Demo the environment
    total_rewards = []
    env = HumanLearningEnvironment(num_particles=1, disable_render=False)
    while True:
        rewards = 0
        done = False
        env.render()
        while not done:
            action = env.control_loop()
            observation, reward, done, info = env.step(action)
            rewards += reward
            if done:
                total_rewards.append(rewards)
                print("Simulation complete. Reward: ", rewards)
                print("Average reward so far: ", np.average(total_rewards))
                env.reset()
            env.render()





