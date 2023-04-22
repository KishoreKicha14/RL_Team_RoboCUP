import gym
from gym import spaces
import numpy as np
import math
import mujoco as mj
import os
import random as random


class Soccer(gym.Env):
    def __init__(self):
        # Define the action space
        # The first action is the angle of rotation (-π to π)
        # The second action is the direction of movement (0: stop, 1: forward)
        self.action_space = spaces.Box(
                                        low=np.array([-np.pi, 0], dtype=np.float32),
                                        high=np.array([np.pi, 1], dtype=np.float32),
                                        dtype=np.float32
                                      )

        # Define the observation space
        # The observation space has 10 dimensions:
        # 1. Ball x-coordinate
        # 2. Ball y-coordinate
        # 3. Ball x-velocity
        # 4. Ball y-velocity
        # 5. Ball angle with respect to x-axis (-pi to pi)
        # 6. Agent x-coordinate
        # 7. Agent y-coordinate
        # 8. Agent x-velocity
        # 9. Agent y-velocity
        # 10. Agent angle with respect to x-axis (-pi to pi)
        low = np.array([-45, -30, -10, -10, -np.pi, -45, -30, -10, -10], dtype=np.float32)
        high = np.array([45, 30, 10, 10, np.pi, 45, 30, 10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.forward=0

        # Configurations
        xml_path = 'mujoco/field.xml' #xml file (assumes this is in the same folder as this file)
        simend = 10 #simulation time
        print_camera_config = 0 #set to 1 to print camera config
                                #this is useful for initializing view of the model)

        # For callback functions
        button_left = False
        button_middle = False
        button_right = False
        lastx = 0
        lasty = 0

        # get the full path
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
        self.data = mj.MjData(self.model)                		# MuJoCo data
        cam = mj.MjvCamera()                        # Abstract camera
        opt = mj.MjvOption()                        # visualization options
        #print(self.model, self.data)
        # initialize visualization data structures
        #mj.mjv_defaultCamera(cam)
        #mj.mjv_defaultOption(opt)
        scene = mj.MjvScene(self.model, maxgeom=10000)
        #context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        self.time=0



        self.boundaries=( "Touch lines1", "Touch lines2", "Touch lines3", "Touch lines4", "Touch lines5", "Touch lines6")
        self.Goal=("Goal lines1", "Goal lines2")

    def reset(self):
        self.time=0
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        print("Reset")
        start_agent_x, start_agent_y, start_ball_x, start_ball_y=self.generate_start_positions()
        self.data.qpos[:2]=[start_agent_x, start_agent_y]
        self.data.qpos[7:9]=[start_ball_x, start_ball_y]
        self.state=np.array([start_agent_x, start_ball_y, 0, 0, 0, start_ball_x, start_ball_y, 0, 0])
        return self.state

    def step(self, action):
        mj.mj_step(self.model, self.data)
        self.time+=1
        angle, speed=action
        direction=self.move_and_rotate(angle)
        direction = np.array(direction[:2])
        direction /= np.linalg.norm(direction)  # normalize the velocity vector
        self.data.qvel[:2] = speed * direction
        reward, goal, foul=self.compute_reward()
        a_pos, b_pos=self.data.xpos[8], self.data.xpos[9]
        agent_x, agent_y, agent_z = a_pos
        ball_x, ball_y, ball_z = b_pos
        #print(data.qvel)
        agent_vx, agent_vy=self.data.qvel[:2]
        ball_vx, ball_vy=self.data.qvel[7:9]
        self.state=np.array([agent_x, agent_y, agent_vx, agent_vy, self.forward, ball_x, ball_y, ball_vx, ball_vy])


        return self.state, reward, self.time==10000-1 or goal, {}

    def render():
        
        pass

    def generate_start_positions(random_val=False):
        if random_val:
            start_agent_x = random.uniform(-45, 45)
            start_agent_y = random.uniform(-30, 30)
            start_ball_x = random.uniform(-45, 45)
            start_ball_y = random.uniform(-30, 30)
        else:
            start_agent_x = 25
            start_agent_y = 0
            start_ball_x = 27
            start_ball_y = 0
            
        return start_agent_x, start_agent_y, start_ball_x, start_ball_y

    def move_and_rotate(self, angle):
        self.forward+=angle
        angle= angle
        x_prime = math.cos(angle)
        y_prime = math.sin(angle)
        z_prime = 0
        return [x_prime, y_prime, z_prime]

    def Is_ball_touched(self):
        for i in range(len(self.data.contact.geom1)):
            if (self.data.geom(self.data.contact.geom1[i]).name == "ball_g" and self.data.geom(self.data.contact.geom2[i]).name == "sphero1") or (self.data.geom(self.data.contact.geom2[i]).name == "ball_g" and self.data.geom(self.data.contact.geom1[i]).name == "sphero1"):
                print("touched_ball")
                return 100
        return 0
    
    def Is_boundaries_touched(self):
        for i in range(len(self.data.contact.geom1)):
            if (self.data.geom(self.data.contact.geom1[i]).name == "sphero1" and self.data.geom(self.data.contact.geom2[i]).name in self.boundaries) or (self.data.geom(self.data.contact.geom2[i]).name == "sphero1" and self.data.geom(self.data.contact.geom1[i]).name in self.boundaries):
                #print("touched_boundary")
                #print(data.xpos[8])
                print("oob")
                return -10000
        return 0
    
    def Is_goal(self):
        for i in range(len(self.data.contact.geom1)):
            if (self.data.geom(self.data.contact.geom1[i]).name == "ball_g" and self.data.geom(self.data.contact.geom2[i]).name in self.Goal) or (self.data.geom(self.data.contact.geom2[i]).name == "ball_g" and self.data.geom(self.data.contact.geom1[i]).name in self.Goal):
                print("Goal!!!")
                return 1000
        return 0
    
    def Is_goal_sphero(self):
        for i in range(len(self.data.contact.geom1)):
            if (self.data.geom(self.data.contact.geom1[i]).name == "sphero1" and self.data.geom(self.data.contact.geom2[i]).name in self.Goal) or (self.data.geom(self.data.contact.geom2[i]).name == "sphero1" and self.data.geom(self.data.contact.geom1[i]).name in self.Goal):
                #print("Sphero Goal!!!")
                return -100
        return 0
    
    def distance_bw_goal1_n_ball(self):
            # define the line by two points a and b
            a = np.array([45, 5, 0])
            b = np.array([45, -5, 0])
            # define the point p
            p = self.data.xpos[8]
            # calculate the distance
            distance = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
            return distance
    
    def distance_bw_goal2_n_ball(self):
            # define the line by two points a and b
            a = np.array([-45, 5, 0])
            b = np.array([-45, -5, 0])
            # define the point p
            p = self.data.xpos[8]
            # calculate the distance
            distance = np.linalg.norm(np.cross(p - a, p - b)) / np.linalg.norm(b - a)
            return distance
    
    def distance_bw_ball_n_sphero(self):
        return np.linalg.norm(self.data.xpos[8] - self.data.xpos[9])

    def compute_reward(self):
        # Compute the distance to the ball and the goal
        distance_to_ball = self.distance_bw_ball_n_sphero()
        #print(distance_to_ball)
        distance_to_goal = self.distance_bw_goal1_n_ball()

        # Compute the time penalty
        time_penalty = -0.0001

        # Compute the out-of-bounds penalty
        out_of_bound_penalty = self.Is_boundaries_touched()
        # Compute the touch ball reward
        touch_ball_reward = self.Is_ball_touched()
        # Compute the goal achieved reward
        goal_achieved_reward = self.Is_goal()

        # Compute the distance to the ball and goal coefficients
        distance_to_goal_coeff = - 0.01
        distance_to_ball_coeff = - 0.01
        Sphero_goal_penalty=self.Is_goal_sphero()

        # Compute the overall reward
        reward = (
                touch_ball_reward +
                goal_achieved_reward +
                distance_to_ball_coeff * distance_to_ball +
                distance_to_goal_coeff * distance_to_goal +
                #time_penalty +
                Sphero_goal_penalty+
                #rotation_penalty +
                out_of_bound_penalty)
        return reward, True if goal_achieved_reward!=0.0 else False, True if out_of_bound_penalty!=0 or Sphero_goal_penalty!=0 else False

