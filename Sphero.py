import mujoco as mj

class Sphero():
	def __init__(self, data, model, name):
		self.name = name
		self.id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.joint_kick = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'kick_' + name)
		self.joint_roll = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, 'roll_' + name)

	def get_position(self, data, model):
		return model.body_pos[self.id, :]

	def set_direction(self, data, x):
		pass

	def set_velocity(self, data):
		pass