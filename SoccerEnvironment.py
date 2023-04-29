import numpy as np
import gym
import torch
from gym import spaces

class SoccerEnvironment(gym.Env):
	def __init__(self, players_per_team, randomize_player_positions = False):
		self.players_per_team = players_per_team
		self.randomize_player_positions = randomize_player_positions

		# Angle of rotation, Direction of movement
		self.action_space = spaces.Box(
			low=np.array([-np.pi, 0], dtype=np.float32),
			high=np.array([np.pi, 1], dtype=np.float32),
			dtype=np.float32
		)

		# x-coord, y-coord, x-vel, y-vel
		information_low_ball = np.array([-45, -30, -10, -10], dtype=np.float32) # TODO: Should increase to boundaries of pitch
		information_high_ball = np.array([45, 30, 10, 10], dtype=np.float32)

		# x-coord, y-coord, x-vel, y-vel, angle wrt x-axis
		information_low_agent = np.array([-45, -30, -10, -10, -np.pi], dtype=np.float32)
		information_high_agent = np.array([45, 30, 10, 10, np.pi], dtype=np.float32)

		number_of_agents = self.players_per_team * 2

		low = np.concatenate((np.tile(information_low_agent, number_of_agents), information_low_ball))
		high = np.concatenate((np.tile(information_high_agent, number_of_agents), information_high_ball))

		self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.lines_boundary=( "Touch lines1", "Touch lines2", "Touch lines3", "Touch lines4", "Touch lines5", "Touch lines6")
		self.lines_goal = ("Goal lines1", "Goal lines2")

	def generate_state_space(self, model, data, players, ball):
		state_space = []
		for player in players:
			state_space += player.get_state(model, data)
		state_space += ball.get_state(model, data)
		state_space = torch.from_numpy(np.array(state_space)).float().unsqueeze(0).to(self.device)
		return state_space
