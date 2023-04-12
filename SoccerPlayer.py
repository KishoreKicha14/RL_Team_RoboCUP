import mujoco as mj

class SoccerPlayer():
	def __init__(self, data, model, name):
		self.name = name
		self.id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)

	def get_position(self, data, model):
		return model.body_pos[self.id, :]

	def set_direction(self, data, x):
		pass

	def set_velocity(self, data):
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