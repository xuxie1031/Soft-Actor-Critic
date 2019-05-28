import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)


class ValueNet(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super(ValueNet, self).__init__()

		self.input2hidden = nn.Linear(input_dim, hidden_dim)
		self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)
		self.hidden2output = nn.Linear(hidden_dim, 1)

		self.apply(weights_init)

	def forward(self, state):
		x = F.relu(self.input2hidden(state))
		x = F.relu(self.hidden2hidden(x))
		x = self.hidden2output(x)

		return x


class QNet(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(QNetwork, self).__init__()

		self.input2hidden0 = nn.Linear(input_dim+action_dim, hidden_dim)
		self.hidden2hidden0 = nn.Linear(hidden_dim, hidden_dim)
		self.hidden2output0 = nn.Linear(hidden_dim, 1)

		self.input2hidden1 = nn.Linear(input_dim+action_dim, hidden_dim)
		self.hidden2hidden1 = nn.Linear(hidden_dim, hidden_dim)
		self.hidden2output1 = nn.Linear(hidden_dim, 1)

		self.apply(weights_init)


	def forward(self, state, action):
		xu = torch.cat([state, action], 1)

		x1 = F.relu(self.input2hidden0(x0))
		x1 = F.relu(self.hidden2hidden0(x1))
		x1 = self.hidden2output0(x1)

		x2 = F.relu(self.input2hidden1(xu))
		x2 = F.relu(self.hidden2hidden1(x2))
		x2 = self.hidden2output1(x2)

		return x1, x2


class GaussianPolicy(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(GaussianPolicy, self).__init__()

		self.input2hidden = nn.Linear(input_dim, hidden_dim)
		self.hidden2hidden = nn.Linear(hidden_dim, hidden_dim)

		self.hidden2mean = nn.Linear(hidden_dim, action_dim)
		self.hidden2logstd = nn.Linear(hidden_dim, action_dim)

		self.apply(weights_init)


	def forward(self, state):
		x = F.relu(self.input2hidden(state))
		x = F.relu(self.hidden2hidden(x))

		mean = self.hidden2mean(x)
		log_std = self.hidden2logstd(x)
		log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)

		return mean, log_std


	def sample(self, state):
		mean, log_std = self.forward(state)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		x_t = normal.rsample()
		action = torch.tanh(x_t)
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log(1-action.pow(2)+epsilon)
		log_prob = log_prob.sum(1, keepdim=True)

		return action, log_prob, torch.tanh(mean)