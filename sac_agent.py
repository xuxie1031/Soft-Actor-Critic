import torch
import torch.nn.functional as F
from models import *


def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)


def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data*(1-tau)+param*tau)


class SACAgent:
	def __init__(self, input_dim, action_dim, args):
		self.gamma = args.gamma
		self.tau = args.tau
		self.alpha = args.alpha

		self.policy_type = args.policy
		self.targetQ_update_interval = args.targetQ_update_interval
		self.auto_entropy_tuning = args.auto_entropy_tuning

		self.device = torch.device('cuda' if args.cuda else 'cpu')

		self.critic = QNet(input_dim, action_dim, args.hidden_dim).to(device=self.device)
		self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

		self.critic_target = QNet(input_dim, action_dim, args.hidden_dim).to(device=self.device)
		hard_update(self.critic_target, self.critic)

		if self.policy_type == 'Gaussian':
			if self.auto_entropy_tuning == True:
				self.target_entropy = -torch.prod(torch.tensor((action_dim, ))).item()
				self.log_alpha = torch.zeros(1, require_grads=True, device=self.device)
				self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

			self.policy = GaussianPolicy(input_dim, action_dim, args.hidden_dim)
			self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=args.lr)


	def select_action(self, state, eval=False):
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		if not eval:
			action, _, _ = self.policy.sample(state)
		else:
			_, _, action = self.policy.sample(state)
		action = action.detach().cpu().numpy()

		return action[0]


	def update_params(self, memory, batch_size, updates):
		batch_state, batch_action, batch_reward, batch_next_state, batch_mask = memory.sample(batch_size=batch_size)

		batch_state = torch.FloatTensor(batch_state).to(self.device)
		batch_action = torch.FloatTensor(batch_action).to(self.device)
		batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
		batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
		batch_mask = torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device)

		with torch.no_grad():
			next_state_action, next_state_log_pi, _ = self.policy.sample(batch_next_state)

			qf0_next_target, qf1_next_target = self.critic_target(batch_next_state, next_state_action)
			min_qf_next_target  = torch.min(qf0_next_target, qf1_next_target)-self.alpha*next_state_log_pi
			next_q = batch_reward+batch_mask*self.gamma*min_qf_next_target

		qf0, qf1 = self.critic(batch_state, batch_action)
		qf0_loss = F.mse_loss(qf0, next_q)
		qf1_loss = F.mse_loss(qf1, next_q)

		pi, log_pi, _ = self.policy.sample(batch_state)

		qf0_pi, qf1_pi = self.critic(batch_state, pi)
		min_qf_pi = torch.min(qf0_pi, qf1_pi)

		policy_loss = (self.alpha*log_pi-min_qf_pi).mean()

		self.critic_optim.zero_grad()
		qf1_loss.backward()
		self.critic_optim.step()

		self.critic_optim.zero_grad()
		qf2_loss.backward()
		self.critic_optim.step()

		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()

		if self.auto_entropy_tuning:
			alpha_loss = -(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()

			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			self.alpha = self.log_alpha.exp()
		else:
			alpha_loss = torch.tensor(0.).to(self.device)

		if updates % self.targetQ_update_interval == 0:
			soft_update(self.critic_target, self.critic, self.tau)

		return qf0_loss.item(), qf1_loss.item(), policy_loss.item(), alpha_loss.item()