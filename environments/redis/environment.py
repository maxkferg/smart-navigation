import sys
import math
import time
import random
import pygame
import pygame.gfxdraw
import matplotlib
import numpy as np
import gym.spaces
from pygame import pixelcopy
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn import preprocessing
from skimage.draw import circle
from .physics.utils import cart2pol
from .physics.universe import Universe
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


class LearningEnvironment:
    """
    Environment that an algorithm can play with
    """
    action_dimensions = 2
    state_dimensions = 5 # The number of dimensions per particle (x,y,theta,tx,ty)
    state_history = 4
    state_buffer = []
    max_steps = 100

    screen = None
    particle_speed = 20
    screen_width = 800
    screen_height = 800
    background = None

    discretization = 8
    spawn = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])


    def __init__(self, num_particles=2, particle_size=50, disable_render=False):
        """
        @history: A vector where each row is a previous state
        """
        self.current_step = 0
        self.num_particles = num_particles
        self.observation_space = ObservationSpace(-1, 1, num_particles * self.state_dimensions * self.state_history + 1)
        self.action_space = ActionSpace(-1, 1, self.action_dimensions)

        self.universe = Universe((self.screen_width, self.screen_height), self.spawn, self.discretization)
        self.universe.addFunctions(['move', 'bounce', 'brownian', 'collide', 'drag'])
        self.universe.mass_of_air = 0.1

        # Add all the particles
        colors = sns.color_palette("muted")
        for i in range(self.num_particles):
            name = "default"
            speed = self.particle_speed
            backend = "simulation"
            if i==0:
                name = "primary"
                speed = self.particle_speed
            color = tuple(255*c for c in colors[i])
            target = self.universe.addTarget(radius=particle_size, color=(0,255,255))
            particle = self.universe.addParticle(radius=particle_size, mass=100, speed=speed, elasticity=0.5, color=color, target=target, name=name)

        # Add the primary particle
        self.primary = self.universe.particles[0]

        # Reset the environment to correctly spawn the particles
        self.reset()
        if not disable_render:
            print('Initializing pygame screen')
            self.screen_buffer = pygame.Surface((self.screen_width, self.screen_height))
            self.screen_buffer.set_alpha(50)
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Bouncing Objects')

    """
    def step(self, action):
        Step forward n steps
        n = 4
        rewards = 0
        for i in range(n):
            state, reward, done, info = self._step(action)
            rewards += reward
            if done:
                break
        return state, rewards, done, info
    """

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
        self.primary.control(steering, throttle)

        # Step forward one timestep
        self.universe.update()
        state = self.get_current_state()
        self.current_step += 1

        if self.primary.atTarget(threshold=50) and self.primary.speed<0.4:
            reward = 1
            done = True
        elif self.primary.collisions > 0:
            reward = -1/self.num_particles
            done = True
        else:
            reward = 0
            done = self.current_step >= self.max_steps

        # Enforce acceleration limits
        excess = np.maximum(abs(action)-0.9, 0)
        reward -= np.sum(excess)

        # Enforce speed limits
        if self.primary.speed > 1:
            reward -= 0.02

        # Enforce penalty regions
        if self.universe.isOnPenalty(self.primary):
            reward -= 0.02

        info = {'step': self.current_step}
        return state, reward, done, info


    def reset(self):
        """Respawn the particles and the targets"""
        self.universe.reset()

        self.state_buffer = []

        self.primary.speed = 0

        self.current_step = 0

        return self.get_current_state()


    def render(self):
        """
        Render the environment
        """
        # Clear the screen
        if self.background is not None:
            pixelcopy.array_to_surface(self.screen, self.background)
        else:
            self.screen.fill(self.universe.colour)

        # Draw penalties
        for xi in range(self.universe.discretization):
            for yi in range(self.universe.discretization):
                xmin,xmax,ymin,ymax = self.universe._get_grid_cell_bounds(xi,yi)
                rect = (xmin, ymin, xmax-xmin, ymax-ymin)
                color = ttl_color(self.universe.penalties[yi,xi])
                pygame.draw.rect(self.screen_buffer, color, rect)
        self.screen.blit(self.screen_buffer, (0,0))

        # Draw Spawn
        #for p in self.universe.spawn:
        #    x,y,w,h = p
        #    rect = (int(x), int(y), int(w), int(h))
        #    color = (200,200,255)
        #    pygame.draw.rect(self.screen, color, rect, 2)

        # Draw particles
        for p in self.universe.particles:
            edge = np.maximum(p.colour, (200,200,200))
            self.draw_circle(int(p.x), int(p.y), p.size, p.colour, edgeColor=edge, filled=True)

        # Draw primary target
        #for t in self.universe.targets:
        t = self.primary.target
        self.draw_circle(int(t.x), int(t.y), t.radius, t.color, filled=False)
        self.draw_circle(int(t.x), int(t.y), int(t.radius/4), t.color, filled=True)

        # Draw the primary particle direction
        p = self.primary
        dx, dy = p.get_speed_vector(scale=100)
        pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (0,0,0))

        # Draw the control vector
        dx, dy = p.get_control_vector(scale=100)
        pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+dx), int(p.y+dy), (255,0,0))

        self.flip_screen()
        time.sleep(0.01)


    def switch_backend(self,backend):
        """Switch the primary backend"""
        self.primary.backend = backend
        pygame.display.set_caption('Bouncing Objects (%s)'%backend)


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


    def get_teacher_action(self):
        """
        Return a chase goal action
        """
        dx = (self.primary.target.x - self.primary.x) / self.screen_width
        dy = (self.primary.target.y - self.primary.y) / self.screen_height
        desired = cart2pol(-dy, dx)[0] # On domain [-pi,pi]
        current = self.primary.angle   # On domain [0, 2*pi]

        # Map current to domain [-pi,pi]
        if current > math.pi:
            current = current - 2*math.pi

        # Calculate the angle update
        da = desired - current

        # Ideal speed. Should incur minimal penaly
        ds = 1

        # Backward left
        if da < -math.pi / 2:
            angle = -(da + math.pi)
            speed = -(ds + self.primary.speed) / 4 # Aim for speed 0.5
        # Backward right
        elif da > math.pi / 2:
            angle = - da + math.pi
            speed = -(0.5 + self.primary.speed) / 4 # Aim for speed 0.5
        # Forward
        else:
            angle = 2 * da # Correct angle
            speed = (0.5 - self.primary.speed) / 4 # Aim for speed 0.5

        speed = clip(speed, -0.9, 0.9)
        angle = clip(angle, -0.9, 0.9)

        return np.array([angle,speed])


    def get_current_state(self):
        """
        Return a representation of the simulation state
        """
        state = []
        for i,particle in enumerate(self.universe.particles):
            state.extend(particle.get_state_vector(self.screen_width, self.screen_height))
        # Append to the state buffer
        while len(self.state_buffer) <= self.state_history:
            self.state_buffer.insert(0,state)
        self.state_buffer.pop()

        positions = self.state_buffer
        #penalties = self.universe.penalties.flatten()/10
        state = sum(positions, [])
        state.append(int(self.primary.backend=="simulation"))
        state = np.array(state)
        return state


    def close(self):
        """Clean up the env"""
        pass



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
                        action = [0, 0.6]
                    if event.key == pygame.K_d:
                        action = [0.3, 0.6]
                    if event.key == pygame.K_s:
                        action = [0, -0.6]
                    if event.key == pygame.K_a:
                        action = [-0.3, 0.6]
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





