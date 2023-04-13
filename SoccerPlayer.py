import mujoco as mj
from DDPG import Actor

class SoccerPlayer():
	def __init__(self, model, data, name, env):
		self.name = name
		self.env = env

		self.id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		
		self.size_hidden_layers = 256
		self.brain = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.size_hidden_layers)

	def get_position(self, model, data):
		return model.body_pos[self.id, :]

	def set_direction(self, model, data, x):
		pass

	def set_velocity(self, model, data):
		pass

	def move_and_rotate(current_coords, angle, forward):
		forward+=angle
		angle= angle
		x, y, z = current_coords
		x_prime = math.cos(angle)
		y_prime = math.sin(angle)
		z_prime = 0
		return  forward, [x_prime, y_prime, z_prime]

	def distance_bw_ball_n_sphero():
		return np.linalg.norm(data.xpos[8] - data.xpos[9])