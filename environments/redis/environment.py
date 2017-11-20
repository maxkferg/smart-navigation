import sys
import math
import time
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
import seaborn as sns; sns.set()


class ObservationSpace(gym.spaces.Box):
    pass


class ActionSpace(gym.spaces.Box):
    pass


def ttl_color(ttl):
    """Return a color based on the ttl"""
    cmap = matplotlib.cm.get_cmap('YlGn')
    if ttl == 0:
        color = (1,0,0)
    else:
        color = cmap(ttl/10)
    return [int(255*color) for color in color]


def clip(val, minimum, maximum):
    """Clip a value to [min,max]"""
    return max(min(val,maximum),minimum)


def clipv(vector, space):
    """Clip @vector to the gym space"""
    return np.clip(vector, space.low, space.high)


def get_color(i):
    """Return a color from the pallete"""
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
    action_dimensions = 2
    state_dimensions = 5 # The number of dimensions per particle (x,y,theta,tx,ty)
    state_history = 4
    state_buffer = []
    max_steps = 100
    record_catastrophies = True

    screen = None
    particle_speed = 20
    screen_width = 800
    screen_height = 800
    background = None
    snapshots = deque(maxlen=10)
    catastrophies = deque(maxlen=1000)


    def __init__(self, primary, num_particles, particle_size, disable_render):
        """
        @history: A vector where each row is a previous state
        """
        self.current_step = 0
        self.reward_so_far = 0
        self.num_particles = num_particles
        self.state_size = num_particles*self.state_dimensions + self.action_dimensions
        self.observation_space = ObservationSpace(-1, 1, shape=(self.state_history, self.state_size))
        self.action_space = ActionSpace(-1, 1, self.action_dimensions)
        self.previous_action = np.zeros(self.action_space.shape)

        self.primary = primary
        self.universe = Universe((self.screen_width, self.screen_height))
        self.universe.addFunctions(['move', 'bounce', 'collide', 'drag', 'move_adversary'])
        self.universe.mass_of_air = 0.1

        # Create the primary
        if primary is not None:
            self.universe.particles.insert(0,primary)
        else:
            name = "primary"
            speed = self.particle_speed
            color = get_color(0)
            target = self.universe.addTarget(radius=particle_size, color=color)
            particle = self.universe.addParticle(radius=particle_size, mass=100, speed=speed, elasticity=0.5, color=color, target=target, name=name)
        self.primary = self.universe.particles[0]

        # Add all the other particles
        for i in range(1,self.num_particles):
            name = "default"
            speed = self.particle_speed
            color = get_color(i)
            target = self.universe.addTarget(radius=particle_size, color=color)
            particle = self.universe.addParticle(radius=particle_size, mass=100, speed=speed, elasticity=0.5, color=color, target=target, name=name)

        # Reset the environment to correctly spawn the particles
        self.reset()
        if not disable_render:
            print('Initializing pygame screen')
            self.screen_buffer = pygame.Surface((self.screen_width, self.screen_height))
            self.screen_buffer.set_alpha(50)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Bouncing Objects')


    def step(self, action):
        """
        Step the environment forward
        Return (observation, reward, done, info)
        """
        # We clip the value even if the learning algorithm chooses not to
        # This should be seen as a last resort, to prevent simulation instability
        action = clipv(action, self.action_space)

        # Particle 1 is being controlled
        steering = action[0]
        throttle = action[1]
        self.previous_action = action
        self.primary.control(steering, throttle)

        # Step forward one timestep
        self.universe.update()

        # Update our primary positions
        self.primary.update()
        self.primary.target.update()

        # Calculate the current state
        state = self.get_current_state()
        self.current_step += 1

        if self.primary.atTarget(threshold=50) and self.primary.speed<0.4:
            reward = 1
            done = True
        elif self.primary.collisions > 0:
            reward = -1
            done = True
        else:
            reward = 0
            done = self.current_step >= self.max_steps

        # Enforce acceleration limits
        excess = np.maximum(abs(action)-0.9, 0)
        reward -= np.sum(excess)

        # Enforce speed limits
        if abs(self.primary.speed) > 1:
            reward -= 0.02

        # Softly penalize reversing
        if self.primary.speed < 0:
            reward -= 0.001

        # Enforce penalty regions
        if self.universe.isOnPenalty(self.primary):
            reward -= 0.1

        # Reward clipping
        self.reward_so_far += reward
        if self.reward_so_far <= -10:
            done = True

        info = {'step': self.current_step}
        return state, reward, done, info


    def reset(self, catastrophy=False):
        """Respawn the particles and the targets"""
        self.universe.reset()

        self.state_buffer = []

        self.primary.speed = 0

        self.current_step = 0

        self.reward_so_far = 0

        return self.get_current_state()


    def render(self):
        """
        Render the environment
        """
        # Clear the screen
        if self.background is not None:
            pixelcopy.array_to_surface(self.screen, self.background)
        else:
            self.screen.fill(self.universe.color)

        # Draw penalties
        for xi in range(self.universe.discretization):
            for yi in range(self.universe.discretization):
                xmin,xmax,ymin,ymax = self.universe._get_grid_cell_bounds(xi,yi)
                rect = (xmin, ymin, xmax-xmin, ymax-ymin)
                color = ttl_color(self.universe.penalties[yi,xi])
                pygame.draw.rect(self.screen_buffer, color, rect)
        self.screen.blit(self.screen_buffer, (0,0))

        # Draw particles
        for p in self.universe.particles:
            edge = np.maximum(p.color, (200,200,200))
            self.draw_circle(int(p.x), int(p.y), p.radius, p.color, edgeColor=edge, filled=True)

        # Draw primary target
        t = self.primary.target
        self.draw_circle(int(t.x), int(t.y), t.radius, t.color, filled=False)
        self.draw_circle(int(t.x), int(t.y), int(t.radius/4), t.color, filled=True)

        # Draw the primary particle speed
        for p in self.universe.particles:
            dx, dy = p.get_speed_vector(scale=20)
            pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (0,0,255))

        # Draw the primary particle orientation
        for p in self.universe.particles:
            dx, dy = p.get_direction_vector(scale=1)
            pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (0,0,0))

        # Draw the control vector
        try:
            p = self.primary
            dx, dy = p.get_control_vector(scale=20)
            pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (255,0,0))
        except AttributeError:
            pass

        self.flip_screen()
        time.sleep(0.01)


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



    def get_current_state(self):
        """
        Return a representation of the simulation state
        The return array is size (self.state_history, self.state_size)
        """
        # Get the current state vector
        state = []
        for i,particle in enumerate(self.universe.particles):
            state.extend(particle.get_state_vector(self.screen_width, self.screen_height))
        state.extend(self.previous_action)

        # Append to the state buffer
        while len(self.state_buffer) <= self.state_history:
            self.state_buffer.insert(0,state)
        self.state_buffer.pop()

        # Convert to numpy array (self.state_history, self.state_size)
        state = np.array(self.state_buffer)
        return state


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
    def __init__(self, num_particles=2, particle_size=80, *args, **kwargs):
        super().__init__(primary, num_particles, particle_size, *args, **kwargs)



class ExecuteEnvironment(Environment):
    """
    A real environment used to control the car
    - The primary position is pulled from the database
    - The primary control signal is sent to the car
    - The target position is pulled from the database
    @locus: The Real Car
    """
    def __init__(self, num_particles=2, particle_size=80, *args, **kwargs):
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
    def __init__(self, num_particles=2, particle_size=80, *args, **kwargs):
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

    def control_loop(self):
        """
        Return a user selected action
        """
        action = None

        while action is None:
            events = pygame.event.get()
            for event in events:
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
    env = HumanLearningEnvironment(num_particles=2, disable_render=False)
    while True:
        rewards = 0
        done = False
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





