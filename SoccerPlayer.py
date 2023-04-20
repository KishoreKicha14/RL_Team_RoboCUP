import mujoco as mj
from DDPG import Actor

class SoccerPlayer():
	def __init__(self, model, data, name, env):
		self.name = name
		self.env = env

		self.id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.id_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
		self.id_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
		
		self.size_hidden_layers = 256
		self.brain = Actor(self.env.observation_space.shape[0], self.env.action_space.shape[0], self.size_hidden_layers)

	def get_pose(self, model, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 7]

	def set_pose(self, model, data, position = None, quaternion = None):
		if position is not None:
			data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position
		if quaternion is not None:
			data.qpos[self.id_joint * 7 + 3: self.id_joint * 7 + 7] = quaternion

	def get_velocity(self, model, data):
		return data.qvel[self.id_joint * 6: self.id_joint * 6 + 6]

	def set_velocity(self, model, data, velocity = None, angular_velocity = None):
		if velocity is not None:
			data.qvel[self.id_joint * 6: self.id_joint * 6 + 3] = velocity
		if angular_velocity is not None:
			data.qvel[self.id_joint * 6 + 3: self.id_joint * 6 + 6] = angular_velocity

	def get_position(self, model, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 3]

	def get_direction(self, model, data):
		return data.qpos[self.id_joint * 7 + 3: self.id_joint * 7 + 7]



	

	def get_action(self, model, data):


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