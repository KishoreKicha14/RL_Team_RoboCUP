import numpy as np

# x-coord, y-coord, x-vel, y-vel, angle wrt x-axis
information_low_ball = np.array([-45, -30, -10, -10, -np.pi], dtype=np.float32) # TODO: should increase to boundaries of pitch
information_high_ball = np.array([45, 30, 10, 10, np.pi], dtype=np.float32)

# x-coord, y-coord, x-vel, y-vel, angle wrt x-axis
information_low_agent = np.array([-45, -30, -10, -10, -np.pi], dtype=np.float32)
information_high_agent = np.array([45, 30, 10, 10, np.pi], dtype=np.float32)

number_of_agents = 3

low = np.concatenate((information_low_ball, np.tile(information_low_agent, number_of_agents)))
high = np.concatenate((information_high_ball, np.tile(information_high_agent, number_of_agents)))

print(low)
print(high)