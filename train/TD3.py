import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
class ActorNetwork(nn.Module):
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.action = nn.Linear(fc2_dim, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
 
    def forward(self, state):
        x = self.fc1(state)
        Leaky_ReLU = torch.nn.LeakyReLU(negative_slope=5e-2)
        x = Leaky_ReLU(self.ln1(x))
        x = Leaky_ReLU(self.ln2(self.fc2(x)))
        action = torch.tanh(self.action(x))
        return action
 
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file=None):
        self.load_state_dict(torch.load(checkpoint_file),strict=False)

 
class CriticNetwork(nn.Module):
    def __init__(self, lr, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
    
    def forward(self, state):
        x = state
        Leaky_ReLU = torch.nn.LeakyReLU(negative_slope=5e-2)
        x = Leaky_ReLU(self.ln1(self.fc1(x)))
        x = Leaky_ReLU(self.ln2(self.fc2(x)))
        q = self.q(x)
        return q
 
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file=None):
        self.load_state_dict(torch.load(checkpoint_file),strict=False)

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0
        self.lowdim_state_memory = np.zeros((max_size, state_dim))
        self.action_memory = np.zeros((max_size, action_dim))
        self.reward_memory = np.zeros((max_size, ))
        self.reward_init_memory = np.zeros((max_size, ))

    def store_transition(self, lowdim_state, action, reward):
        mem_idx = self.mem_cnt % self.mem_size
        self.lowdim_state_memory[mem_idx] = lowdim_state
        self.action_memory[mem_idx] = action
        self.reward_memory[mem_idx] = reward
        self.mem_cnt += 1
        
    def sample_buffer(self):
        mem_len = min(self.mem_cnt, self.mem_size)
        batch = np.random.choice(mem_len, self.batch_size, replace=False)
        lowdim_states = self.lowdim_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        return lowdim_states, actions, rewards
 
    def ready(self):
        return self.mem_cnt >= self.batch_size

class TD3:
    def __init__(self, lr_actor, lr_critic, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, learn_time, ckpt_dir, gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=10000000,
                 batch_size=256):
        self.LEARN_TIME = learn_time
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir
        self.actor = ActorNetwork(lr=lr_actor, state_dim=state_dim, action_dim=action_dim, fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic1 = CriticNetwork(lr=lr_critic, state_dim=state_dim, action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_actor = ActorNetwork(lr=lr_actor, state_dim=state_dim, action_dim=action_dim, fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_critic1 = CriticNetwork(lr=lr_critic, state_dim=state_dim, action_dim=action_dim, fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim, batch_size=batch_size)
        self.update_network_parameters(tau=1.0)
    
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        for actor_params, target_actor_params in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)
        for critic1_params, target_critic1_params in zip(self.critic1.parameters(), self.target_critic1.parameters()):
             target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)
    
    def remember(self, lowdim_state, action, reward_sum):
        self.memory.store_transition(lowdim_state, action, reward_sum)
 
    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = torch.tensor([observation], dtype=torch.float).to(device)
        action = self.actor.forward(state)
        if train:
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise, size=3), dtype=torch.float).to(device)
            action = torch.clamp(action+noise, -1, 1)
        self.actor.train()
        return action.squeeze().detach().cpu().numpy()
 
    def learn(self):
        if not self.memory.ready():
            return
        all_critic_loss = 0.0
        all_actor_loss = 0.0
        for i in range(self.LEARN_TIME):
            lowdim_states, actions, rewards = self.memory.sample_buffer()
            lowdim_states_tensor = torch.tensor(lowdim_states, dtype=torch.float).to(device)
            actions_tensor = torch.tensor(actions, dtype=torch.float).to(device)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float).reshape(-1,1).to(device)
            q1 = self.critic1.forward(torch.cat([lowdim_states_tensor, actions_tensor],dim=-1))
            critic_loss = F.mse_loss(q1, rewards_tensor)
            self.critic1.optimizer.zero_grad()
            critic_loss.backward()
            self.critic1.optimizer.step()
            self.update_time += 1
            new_actions_tensor = self.actor.forward(lowdim_states_tensor)
            q1 = self.critic1.forward(torch.cat([lowdim_states_tensor, new_actions_tensor],dim=-1))
            actor_loss = -torch.mean(torch.sum(q1,dim=1))
            all_critic_loss += critic_loss
            all_actor_loss += actor_loss
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            self.update_network_parameters()
        print('critic loss:',all_critic_loss/self.LEARN_TIME)
        print('actor loss:',all_actor_loss/self.LEARN_TIME)
        return (all_actor_loss/self.LEARN_TIME).item(),(all_critic_loss/self.LEARN_TIME).item()
 
    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Saving actor network successfully!')
        self.target_actor.save_checkpoint(self.checkpoint_dir +'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Saving target_actor network successfully!')
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Saving critic1 network successfully!')
        self.target_critic1.save_checkpoint(self.checkpoint_dir +'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Saving target critic1 network successfully!')
 
    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        print('Loading actor network successfully!')
        self.target_actor.load_checkpoint(self.checkpoint_dir + 'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        print('Loading target_actor network successfully!')
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        print('Loading critic1 network successfully!')
        self.target_critic1.load_checkpoint(self.checkpoint_dir +'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        print('Loading target critic1 network successfully!')
