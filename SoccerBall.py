import mujoco as mj
import numpy as np

class SoccerBall():
	def __init__(self, model, data, name):
		self.name = name

		self.id_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
		self.id_geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
		self.id_joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)

		self._last_hit = None
		self._hit = False
		self._repossessed = False
		self._intercepted = False

		# Tracks distance traveled by the ball in between consecutive hits.
		self._pos_at_last_step = None
		self._dist_since_last_hit = None
		self._dist_between_last_hits = None

	def get_position(self, model, data):
		return data.qpos[self.id_joint * 7: self.id_joint * 7 + 3]

	def set_position(self, model, data, position):
		data.qpos[self.id_joint * 7: self.id_joint * 7 + 3] = position

	def get_velocity(self, model, data):
		return data.qvel[self.id_joint * 6: self.id_joint * 6 + 6]

	def is_ball_touched(self, data, model):
		for c in contacts:
			if c.geom1 == ball_id and c.geom2 in player_ids:
				player_id = c.geom2
				last_touch[player_id] = player_id
				print(f"Player {player_ids.index(player_id) + 1} touched the ball!")

	def Is_boundaries_touched():
		boundaries=( "Touch lines1", "Touch lines2", "Touch lines3", "Touch lines4", "Touch lines5", "Touch lines6",)
		for i in range(len(data.contact.geom1)):
			if (data.geom(data.contact.geom1[i]).name == "sphero1" and data.geom(data.contact.geom2[i]).name in boundaries) or (data.geom(data.contact.geom2[i]).name == "sphero1" and data.geom(data.contact.geom1[i]).name in boundaries):
				#print("touched_boundary")
				#print(data.xpos[8])
				return -10000
		return 0

	Goal=("Goal lines1", "Goal lines2")
	def Is_goal():
		for i in range(len(data.contact.geom1)):
			if (data.geom(data.contact.geom1[i]).name == "ball_g" and data.geom(data.contact.geom2[i]).name in Goal) or (data.geom(data.contact.geom2[i]).name == "ball_g" and data.geom(data.contact.geom1[i]).name in Goal):
				#print("Goal!!!")
				return 1000
		return 0

	def Is_goal_sphero():
		for i in range(len(data.contact.geom1)):
			if (data.geom(data.contact.geom1[i]).name == "sphero1" and data.geom(data.contact.geom2[i]).name in Goal) or (data.geom(data.contact.geom2[i]).name == "sphero1" and data.geom(data.contact.geom1[i]).name in Goal):
				#print("Sphero Goal!!!")
				return -100
		return 0

	def distance_bw_goal1_n_ball():
		# define the line by two points a and b
		a = np.array([45, 5, 0])
		b = np.array([45, -5, 0])
		# define the point p
		p = data.xpos[8]
		# calculate the distance
		distance = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
		return distance

	def distance_bw_goal2_n_ball():
		# define the line by two points a and b
		a = np.array([-45, 5, 0])
		b = np.array([-45, -5, 0])
		# define the point p
		p = data.xpos[8]
		# calculate the distance
		distance = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
		return distance

	def reset(self, model, data):
		self._last_hit = None
		self._hit = False
		self._repossessed = False
		self._intercepted = False

		self._pos_at_last_step = None
		self._dist_since_last_hit = None
		self._dist_between_last_hits = None

		self.set_position(model, data, (0, 0, 0.365))

	def get_state(self, model, data):
		return np.concatenate((self.get_position(model, data)[:2], self.get_velocity(model, data)[:2])).tolist()