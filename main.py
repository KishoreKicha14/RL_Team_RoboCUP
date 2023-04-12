from Simulation import Simulation

envFile = 'SoccerEnvironment.xml'
simTime = 3 # seconds

for i in range(0,2):
	sim = Simulation(envFile, simTime)
	sim.start()