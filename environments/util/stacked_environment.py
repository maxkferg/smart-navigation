import collections
import numpy as np
from .spaces import ObservationSpace, prepend_dimension


class StackedEnvWrapper():

	def __init__(self, env, state_history_len):
		"""Add the state_history dimension"""
		self.env = env
		self.state_history_len = state_history_len
		self.history = collections.deque(maxlen=self.state_history_len)
		state_shape = prepend_dimension(env.observation_space.shape, self.state_history_len)
		# Copy over the standard properties from the wrapped environment
		self.observation_space = ObservationSpace(-1, 1, shape=state_shape)
		self.action_space = env.action_space
		self.reward_range = env.reward_range
		self.screen_height = env.screen_height
		self.screen_width = env.screen_width
		self.metadata = env.metadata
		self.spec = env.spec

	def render(self, *args, **kwargs):
		return self.env.render(*args,**kwargs)

	def step(self, *args, **kwargs):
		"""Return the state history, not just the current state"""
		state, reward, done, info = self.env.step(*args, **kwargs)
		self.history.append(state)
		state = self.get_current_state()
		return state, reward, done, info

	def reset(self, *args, **kwargs):
		"""Reset the history as well"""
		state = self.env.reset(*args, **kwargs)
		for i in range(self.state_history_len):
			self.history.append(state)
		return self.get_current_state()

	def get_current_state(self,*args,**kwargs):
		"""Return the current state, including state history"""
		return np.array(self.history)

	def close(self, *args, **kwargs):
		return self.env.close()