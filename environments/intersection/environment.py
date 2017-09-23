import sys
import math
import time
import random
import pygame
import pygame.gfxdraw
import numpy as np
from pygame import pixelcopy
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense
from sklearn import preprocessing
from skimage.draw import circle
from .physics.utils import cart2pol
from .physics.universe import Universe
import seaborn as sns; sns.set()



class ObservationSpace:

    def __init__(self, state_size):
        self.shape = (state_size,)


class ActionSpace:

    def __init__(self, num_particles, action_dimensions):
        self.shape = (action_dimensions,)



class LearningEnvironment:
    """
    Environment that an algorithm can play with
    """
    action_dimensions = 2
    state_dimensions = 6 # The number of dimensions per particle (x,y,dx,dy,tx,ty)
    max_steps = 400

    screen = None
    particle_speed = 20
    screen_width = 800
    screen_height = 800
    background = None


    def __init__(self, num_particles=4, particle_size=30, disable_render=False):
        """
        @history: A vector where each row is a previous state
        """
        self.current_step = 0
        self.num_particles = num_particles
        self.observation_space = ObservationSpace(num_particles * self.state_dimensions)
        self.action_space = ActionSpace(num_particles, self.action_dimensions)

        self.universe = Universe((self.screen_width, self.screen_height))
        self.universe.addFunctions(['move', 'bounce', 'brownian', 'collide', 'drag'])
        self.universe.mass_of_air = 0.001

        # Add the penalties
        xs = self.screen_width
        ys = self.screen_height
        self.universe.addPenalty(0, 0, xs/4, ys/4)
        self.universe.addPenalty(3/4*xs, 1/4*ys, xs, 0)
        self.universe.addPenalty(1/4*xs, 3/4*ys, 0, ys)
        self.universe.addPenalty(3/4*xs, 3/4*ys, xs, ys)

        # Add the spawn regions (Top left corner. Width and height)
        self.universe.addSpawn(1/4*xs, 0, xs/4, ys/4)
        self.universe.addSpawn(1/2*xs, 0, xs/4, ys/4)
        self.universe.addSpawn(0, 1/4*ys, xs/4, ys/4)
        self.universe.addSpawn(0, 2/4*ys, xs/4, ys/4)
        self.universe.addSpawn(1/4*xs, 3/4*ys, xs/4, ys/4)
        self.universe.addSpawn(1/2*xs, 3/4*ys, xs/4, ys/4)
        self.universe.addSpawn(3/4*xs, 1/4*ys, xs/4, ys/4)
        self.universe.addSpawn(3/4*xs, 1/2*ys, xs/4, ys/4)


        # Add all the particles
        colors = sns.color_palette("muted")
        for i in range(self.num_particles):
            name = "primary" if i==0 else "default"
            speed = 0 if i==0 else self.particle_speed
            color = tuple(255*c for c in colors[i])
            target = self.universe.addTarget(radius=particle_size, color=(0,255,255))
            particle = self.universe.addParticle(radius=particle_size, mass=100, speed=speed, elasticity=0.5, color=color, target=target, name=name)

        # Add the primary particle
        self.primary = self.universe.particles[0]

        # Reset the environment to correctly spawn the particles
        self.reset()

        if not disable_render:
            print('Initializing pygame screen')
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Bouncing Objects')


    def step(self, action, n):
        """
        Step forward n steps
        """
        rewards = 0
        for i in range(n):
            state, reward, done, info = self._step(action)
            rewards += reward
            if done:
                break
        return state, rewards, done, info


    def _step(self, action):
        """
        Step the environment forward
        Return (observation, reward, done, info)
        """
        # Particle 1 is being controlled
        angle, accel = cart2pol(-action[1],action[0])
        self.primary.accelerate((angle, 2*accel))

        # Step forward one timestep
        collisions = self.universe.update()
        state = self.get_current_state()
        self.current_step += 1

        if self.primary.atTarget(threshold=40):
            reward = 1
            done = True
        elif collisions > 0:
            reward = -1/self.num_particles
            done = True
        elif self.universe.isOnPenalty(self.primary):
            reward = -0.1
            done = self.current_step >= self.max_steps
        else:
            reward = 0
            done = self.current_step >= self.max_steps

        excess = np.maximum(abs(action)-0.9, 0)
        reward -= np.sum(excess)

        info = {'step': self.current_step}
        return state, reward, done, info


    def reset(self):
        """Respawn the particles and the targets"""
        self.universe.reset()

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
        for p in self.universe.penalties:
            x1,y1,x2,y2 = p
            rect = (int(min(x1,x2)), int(min(y1,y2)), int(abs(x1-x2)), int(abs(y1-y2)))
            color = (255,200,200)
            pygame.draw.rect(self.screen, color, rect, 2)

        # Draw Spawn
        for p in self.universe.spawn:
            x,y,w,h = p
            rect = (int(x), int(y), int(w), int(h))
            color = (200,200,255)
            pygame.draw.rect(self.screen, color, rect, 2)

        # Draw particles
        for p in self.universe.particles:
            edge = np.maximum(p.colour, (200,200,200))
            self.draw_circle(int(p.x), int(p.y), p.size, p.colour, edgeColor=edge, filled=True)

        # Draw primart target
        #for t in self.universe.targets:
        t = self.primary.target
        self.draw_circle(int(t.x), int(t.y), t.radius, t.color, filled=False)
        self.draw_circle(int(t.x), int(t.y), int(t.radius/4), t.color, filled=True)

        # Draw the primary particle direction
        p = self.primary
        dx, dy = p.get_speed_vector()
        pygame.gfxdraw.line(self.screen, int(p.x), int(p.y), int(p.x+10*dx), int(p.y+10*dy), (0,0,0))
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


    def get_default_action(self):
        """
        Return a chase goal action
        """
        dx = (self.primary.target.x - self.primary.x) / self.screen_width
        dy = (self.primary.target.y - self.primary.y) / self.screen_height

        return np.array([dx,dy])


    def get_current_state(self):
        """
        Return a representation of the simulation state
        """
        state = []
        for i,particle in enumerate(self.universe.particles):
            state.extend(particle.get_state_vector(self.screen_width, self.screen_height))
        return np.array(state)


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
                        action = [0, -0.3]
                    if event.key == pygame.K_d:
                        action = [0.3, 0]
                    if event.key == pygame.K_s:
                        action = [0, 0.3]
                    if event.key == pygame.K_a:
                        action = [-0.3, 0]
        return np.array(action)


if __name__=="__main__":
    # Demo the environment
    total_rewards = []
    env = HumanLearningEnvironment(num_particles=4, disable_render=False)
    while True:
        rewards = 0
        done = False
        while not done:
            env.render()
            action = env.control_loop()
            observation, reward, done, info = env.step(action, n=4)
            rewards += reward
            if done:
                total_rewards.append(rewards)
                print("Simulation complete. Reward: ",rewards)
                print("Average reward so far: ",np.average(total_rewards))
                env.reset()
        env.render()




