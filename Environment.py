from gym import spaces
import numpy as np

class Environment:
	def __init__(self):
		self.action_space = spaces.Box(
			low  = np.array([-np.pi, 0], dtype = np.float32),
			high = np.array([np.pi, 1], dtype = np.float32),
			dtype=np.float32
		)