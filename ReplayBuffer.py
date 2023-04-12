# Replay buffer
class ReplayBuffer:
	def __init__(self):
		self.buffer = []
		self.max_size = BUFFER_SIZE
		self.ptr = 0
		
	def add(self, state, action, reward, next_state, done):
		if len(self.buffer) < self.max_size:
			self.buffer.append(None)
		self.buffer[self.ptr] = (state, action, reward, next_state, done)
		self.ptr = (self.ptr + 1) % self.max_size
		
	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward.reshape(-1, 1), next_state, done.reshape(-1, 1)