import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
from gym.wrappers import GrayScaleObservation
from gym.wrappers import ResizeObservation
from gym.wrappers import FrameStack
import random

class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)
		### TODO ###
		# env = gym.make(config["env_id"], render_mode='rgb_array')
		env = gym.make(config["env_id"], render_mode='human')

		env = ResizeObservation(env, 84)
		env = GrayScaleObservation(env)
	
		env = FrameStack(env, 4)
		
		#print(env.observation_space)
		# self.seed = 40
		# torch.manual_seed(self.seed)
		# random.seed(self.seed)
		# np.random.seed(self.seed)
		# env.seed(self.seed)

		self.env = env
		self.test_env = env
		

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())
		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

		



	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None):
		### TODO ###
		# get action from behavior net, with epsilon-greedy selection
		

		if random.random() < epsilon:
			action = action_space.sample()
		else:
			
			action = self.behavior_net(torch.tensor(np.asarray(
				observation), dtype=torch.float).view(1, 4, 84, 84).to(self.device)).max(1)[1].item()
			#print(action)
		return action

		


	
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)
		
		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		#print(action)
		q_value = self.behavior_net(state).gather(1, action.to(torch.int64))

		with torch.no_grad():
			q_next = self.target_net(next_state).max(1)[0].view(-1, 1)
			#print(q_next)
			# if episode terminates at next_state, then q_target = reward
			# torch.tensor(done, dtype = torch.int64)
			# done: {0, 1}
			
			q_target = q_next * self.gamma * (1 - done) + reward
			#print("q_target", q_target.shape)
			# print("state", state.shape)
			# print("action", action.shape)
			# print("reward", reward.shape)

			# print("done", done.shape)
			# print("q_next", q_next.shape)
			# print("q_target", q_target.shape)
			
		
		criterion = nn.SmoothL1Loss()
		# criterion = nn.MSELoss()
		loss = criterion(q_value, q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	
	
