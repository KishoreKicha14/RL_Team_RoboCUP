# Critic network
class Critic(nn.Module):
	def __init__(self, state_size, action_size, hidden_size):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(state_size + action_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)
		self.relu = nn.ReLU()
		
	def forward(self, state, action):
		x = torch.cat([state, action], 1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x