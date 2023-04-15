from Simulation import Simulation

env_file = 'SoccerEnvironment.xml'
sim_time = 3 # seconds
players_per_team = 2

sim = Simulation(env_file, sim_time, players_per_team)
for i in range(0,3):
	sim.start(i)
sim.stop()