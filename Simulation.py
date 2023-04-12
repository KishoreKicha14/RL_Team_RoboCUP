import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

class Simulation:
	def __init__(self, envFile, simTime):
		self.env_path = None
		self.set_env_path(envFile)

		self.simend = simTime

		self.button_left = False
		self.button_middle = False
		self.button_right = False
		self.lastx = 0
		self.lasty = 0

		self.model = mj.MjModel.from_xml_path(self.env_path) 
		self.data = mj.MjData(self.model)                		
		self.cam = mj.MjvCamera()                       
		self.opt = mj.MjvOption()

		glfw.init()
		self.window = glfw.create_window(1200, 900, 'Sphero Soccer', None, None)
		glfw.make_context_current(self.window)
		glfw.swap_interval(1)

		mj.mjv_defaultCamera(self.cam)
		mj.mjv_defaultOption(self.opt)
		self.scene = mj.MjvScene(self.model, maxgeom=10000)
		self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)                

		glfw.set_key_callback(self.window, self.keyboard)
		glfw.set_cursor_pos_callback(self.window, self.mouse_move)
		glfw.set_mouse_button_callback(self.window, self.mouse_button)
		glfw.set_scroll_callback(self.window, self.mouse_scroll)	

	def set_env_path(self, envFile):
		dirname = os.path.dirname(__file__)
		abspath = os.path.join(dirname + '/' + envFile)
		self.env_path = abspath

	def keyboard(self, window, key, scancode, act, mods):
		if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
			mj.mj_resetData(self.model, self.data)
			mj.mj_forward(self.model, self.data)

	def mouse_button(self, window, button, act, mods):
		self.button_left = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
		self.button_middle = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
		self.button_right = (glfw.get_mouse_button(
			window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

		# update mouse position
		glfw.get_cursor_pos(window)

	def mouse_move(self, window, xpos, ypos):
		dx = xpos - self.lastx
		dy = ypos - self.lasty
		self.lastx = xpos
		self.lasty = ypos

		# no buttons down: nothing to do
		if (not self.button_left) and (not self.button_middle) and (not self.button_right):
			return

		# get current window size
		width, height = glfw.get_window_size(window)

		# get shift key state
		PRESS_LEFT_SHIFT = glfw.get_key(
			window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
		PRESS_RIGHT_SHIFT = glfw.get_key(
			window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
		mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

		# determine action based on mouse button
		if self.button_right:
			if mod_shift:
				action = mj.mjtMouse.mjMOUSE_MOVE_H
			else:
				action = mj.mjtMouse.mjMOUSE_MOVE_V
		elif self.button_left:
			if mod_shift:
				action = mj.mjtMouse.mjMOUSE_ROTATE_H
			else:
				action = mj.mjtMouse.mjMOUSE_ROTATE_V
		else:
			action = mj.mjtMouse.mjMOUSE_ZOOM

		mj.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

	def mouse_scroll(self, window, xoffset, yoffset):
		action = mj.mjtMouse.mjMOUSE_ZOOM
		mj.mjv_moveCamera(self.model, action, 0.0, -0.05 * yoffset, self.scene, self.cam)

	def init_controller(self, model, data):
		pass

	def controller(self, model, data):
		pass

	def start(self):
		self.init_controller(self.model, self.data)
		mj.set_mjcb_control(self.controller)

		while not glfw.window_should_close(self.window):
			time_prev = self.data.time

			while (self.data.time - time_prev < 1.0/60.0):
				mj.mj_step(self.model, self.data)

			if (self.data.time >= self.simend):
				break

			# get framebuffer viewport
			viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
			viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

			# Update scene and render
			mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
			mj.mjr_render(viewport, self.scene, self.context)

			# swap OpenGL buffers (blocking call due to v-sync)
			glfw.swap_buffers(self.window)

			# process pending GUI events, call GLFW callbacks
			glfw.poll_events()

		glfw.terminate()

