import numpy as np
import gym
from gym import spaces

class SoccerEnvironment(gym.Env):
	def __init__(self, players_per_team, randomize_player_positions = False):
		self.players_per_team = players_per_team
		self.randomize_player_positions = randomize_player_positions

		self.action_space = spaces.Box(
			low=np.array([-np.pi, 0], dtype=np.float32),
			high=np.array([np.pi, 1], dtype=np.float32),
			dtype=np.float32
		)

		# x-coord, y-coord, x-vel, y-vel, angle wrt x-axis
		information_low_ball = np.array([-45, -30, -10, -10, -np.pi], dtype=np.float32) # TODO: Should increase to boundaries of pitch
		information_high_ball = np.array([45, 30, 10, 10, np.pi], dtype=np.float32)

		# x-coord, y-coord, x-vel, y-vel, angle wrt x-axis
		information_low_agent = np.array([-45, -30, -10, -10, -np.pi], dtype=np.float32)
		information_high_agent = np.array([45, 30, 10, 10, np.pi], dtype=np.float32)

		number_of_agents = self.players_per_team * 2

		low = np.concatenate((information_low_ball, np.tile(information_low_agent, number_of_agents)))
		high = np.concatenate((information_high_ball, np.tile(information_high_agent, number_of_agents)))

		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

		self.lines_boundary=( "Touch lines1", "Touch lines2", "Touch lines3", "Touch lines4", "Touch lines5", "Touch lines6")
		self.lines_goal = ("Goal lines1", "Goal lines2")