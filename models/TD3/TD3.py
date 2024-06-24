from models.TD3.model import Actor, Critic
from models.TD3.utils import ReplayBuffer
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import active_dir, active_dir_optim

#implementation from paper: https://arxiv.org/pdf/1802.09477.pdf
#source: https://github.com/sfujim/TD3/blob/master/TD3.py

#Set to cuda (gpu) instance if compute available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    """Agent that plays and learn from experience. Hyper-paramters chosen from paper."""
    def __init__(
            self, 
            state_size, 
            action_size, 
            max_action, 
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.1,
            policy_freq=2
        ):
        """
        Initializes the Agent.
        @Param:
        1. state_size: env.observation_space.shape[0]
        2. action_size: env.action_size.shape[0]
        3. max_action: list of max values that the agent can take, i.e. abs(env.action_space.high)
        4. discount: return rate
        5. tau: soft target update
        6. policy_noise: noise reset level, DDPG uses Ornstein-Uhlenbeck process
        7. noise_clip: sets boundary for noise calculation to prevent from overestimation of Q-values
        8. policy_freq: number of timesteps to update the policy (actor) after
        """
        super(Agent, self).__init__()

        #Actor Network initialization
        self.actor = Actor(state_size, action_size, max_action).to(device)
        self.actor.apply(self.init_weights)
        self.actor_target = copy.deepcopy(self.actor) #loads main model into target model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        #Critic Network initialization
        self.critic = Critic(state_size, action_size).to(device)
        self.critic.apply(self.init_weights)
        self.critic_target = copy.deepcopy(self.critic) #loads main model into target model
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def init_weights(self, layer, gain=1):
        # -------------------------------------------
        # # case 1
        # """Xaviar Initialization of weights"""
        # if(type(layer) == nn.Linear):
        #   nn.init.xavier_normal_(layer.weight)
        #   layer.bias.data.fill_(0.01)
        # -------------------------------------------
        # -------------------------------------------
        # case 2
        """
        Orthogonal initialization (used in PPO and A2C)
        """
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                layer.bias.data.fill_(0.0)
        # -------------------------------------------

    def select_action(self, state, action_space):
        """Selects an automatic epsilon-greedy action based on the policy"""
        # -------------------------------------------
        # # case 1
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # unscaled_action = self.actor(state).cpu().data.numpy().flatten()
        # scaled_action = self.scale_action(unscaled_action, action_space)
        # -------------------------------------------
        # case 2 [this works better]
        unscaled_action = action_space.sample()  # sample() is a feature in gym environments
        scaled_action = self.scale_action(unscaled_action, action_space)
        # -------------------------------------------
        return scaled_action

    def scale_action(self, action: np.ndarray, action_space) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = action_space.low, action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray, action_space) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = action_space.low, action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    
    def train(self, replay_buffer:ReplayBuffer):
        """Train the Agent"""

        self.total_it += 1

        # Sample replay buffer 
        state, action, reward, next_state, done = replay_buffer.sample()#sample 256 experiences

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise #noise only set in training to prevent from overestimation
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action) #Q1, Q2
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q #TD-target

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action) #Q1, Q2

        # Compute critic loss using MSE
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates (DDPG baseline = 1)
        if(self.total_it % self.policy_freq == 0):

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state))[0].mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update by updating the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        """Saves the Actor Critic local and target models"""
        torch.save(self.critic.state_dict(), active_dir + filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), active_dir_optim + filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), active_dir + filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), active_dir_optim + filename + "_actor_optimizer")


    def load(self, filename):
        """Loads the Actor Critic local and target models"""
        self.critic.load_state_dict(torch.load(active_dir + filename + "_critic", map_location='cpu'))
        self.critic_optimizer.load_state_dict(torch.load(active_dir_optim + filename + "_critic_optimizer", map_location='cpu'))#optional
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(active_dir + filename + "_actor", map_location='cpu'))
        self.actor_optimizer.load_state_dict(torch.load(active_dir_optim + filename + "_actor_optimizer", map_location='cpu'))#optional
        self.actor_target = copy.deepcopy(self.actor)
