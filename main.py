from Simulation import Simulation

env_file = 'SoccerEnvironment.xml'
sim_time = 3 # seconds
players_per_team = 2

for i in range(0,2):
	sim = Simulation(env_file, sim_time, players_per_team)
	sim.start()