import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import multivariate_normal, normal
from torch.nn import init


class NAF_network(nn.Module):
        def __init__(self, state_dim, action_dim, action_low, action_high, device = 'cpu'):
            super(NAF_network, self).__init__()
            
            self.device = device

            self.sharefc1 = nn.Linear(state_dim, 64)
            self.sharefc2 = nn.Linear(64, 64)
            
            self.v_fc1 = nn.Linear(64, 1)
            
            self.miu_fc1 = nn.Linear(64, action_dim)
            
            self.L_fc1 = nn.Linear(64, action_dim ** 2)
            
            self.action_dim = action_dim
            self.action_low, self.action_high = action_low, action_high

            
        def forward(self, s, a = None):
            
            s = F.relu(self.sharefc1(s))
            # s = F.relu(self.sharefc2(s))
            
            v = self.v_fc1(s)
            
            miu = self.miu_fc1(s)
            
            # currently could only clip according to the same one single value.
            # but different dimensions may mave different high and low bounds
            # modify to clip along different action dimension
            miu = torch.clamp(miu, self.action_low, self.action_high)
            
            if a is None:
                return v, miu
        
            L = self.L_fc1(s)
            L = L.view(-1, self.action_dim, self.action_dim)
            
            tril_mask = torch.tril(torch.ones(
             self.action_dim, self.action_dim, device = self.device), diagonal=-1).unsqueeze(0)
            diag_mask = torch.diag(torch.diag(
             torch.ones(self.action_dim, self.action_dim, device = self.device))).unsqueeze(0)

            L = L * tril_mask.expand_as(L) +  torch.exp(L * diag_mask.expand_as(L))    
            # L = L * tril_mask.expand_as(L) + L ** 2 * diag_mask.expand_as(L)
            
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (a - miu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]
            
            q = A + v
            
            return q
        
        
class DQN_fc_network(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_layers):
            super(DQN_fc_network, self).__init__()
            
            self.fc_in = nn.Linear(input_dim, 32)
            self.fc_hiddens = [nn.Linear(32,32) for i in range(hidden_layers)]
            self.fc_out = nn.Linear(32, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc_in(x))
            for layer in self.fc_hiddens:
                x = F.relu(layer(x))
            x = self.fc_out(x)
            return x
        
class DQN_dueling_network(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_layers):
            super(DQN_dueling_network, self).__init__()
            self.fc_in = nn.Linear(input_dim, 32)
            self.fc_hiddens = [nn.Linear(32,32) for i in range(hidden_layers - 1)]
            
            self.fca_before = nn.Linear(32, 16)
            self.fcv_before = nn.Linear(32, 16)
            self.fca = nn.Linear(16, output_dim)
            self.fcv = nn.Linear(16, 1)
            
        def forward(self, x):
            x = F.relu(self.fc_in(x))
            
            for layer in self.fc_hiddens:
                x = F.relu(layer(x))
            
            a = F.relu(self.fca_before(x))
            a = self.fca(a)
            a -= a.mean()
            v = F.relu(self.fcv_before(x))
            v = self.fcv(v)
            q = a + v
            return q        

class DDPG_critic_network(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        
        super(DDPG_critic_network, self).__init__()
        
        self.sfc1 = nn.Linear(state_dim, 32)
        # self.sfc2 = nn.Linear(64,32)
        
        self.afc1 = nn.Linear(action_dim, 32)
        # self.afc2 = nn.Linear(64,32)
        
        self.sharefc1 = nn.Linear(64,64)
        self.sharefc2 = nn.Linear(64,1)
        
    def forward(self, s, a):
        s = F.relu(self.sfc1(s))
        # s = F.relu(self.sfc2(s))
        
        a = F.relu(self.afc1(a))
        # a = F.relu(self.afc2(a))
        
        qsa = torch.cat((s,a), 1)
        qsa = F.relu(self.sharefc1(qsa))
        qsa = self.sharefc2(qsa)
        
        return qsa
    
class DDPG_actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        
        super(DDPG_actor_network, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
        self.action_low, self.action_high = action_low, action_high
        
    def forward(self, s):
        
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        a = self.fc3(s)
        a = a.clamp(self.action_low, self.action_high)
        
        return a
    
class AC_v_fc_network(nn.Module):
    
    def __init__(self, state_dim):
        super(AC_v_fc_network, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,1)
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        v = self.fc3(s)
        
        return v
    
class AC_a_fc_network(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(AC_a_fc_network, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            return F.softmax(x, dim = 1)
        
class CAC_a_fc_network(nn.Module):
    def __init__(self, input_dim, output_dim, action_low = -1.0, action_high = 1.0, sigma = 1, device = 'cpu'):
        super(CAC_a_fc_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        self.sigma = torch.ones((output_dim), device = device) * sigma
        self.action_low, self.action_high = action_low, action_high
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        mu = self.fc3(s)
        mu = torch.clamp(mu, self.action_low, self.action_high)
        
        # m = multivariate_normal.MultivariateNormal(loc = mu, covariance_matrix= self.sigma)
        m = normal.Normal(loc = mu, scale= self.sigma)

        return m


class CAC_a_sigma_fc_network(nn.Module):
    def __init__(self, input_dim, output_dim, action_low = -1.0, action_high = 1.0, sigma = 1):
        super(CAC_a_sigma_fc_network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fcmu = nn.Linear(64, output_dim)
        self.fcsigma = nn.Linear(64, output_dim)
        
        self.action_low, self.action_high = action_low, action_high
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        # s = F.relu(self.fc2(s))
        mu = self.fcmu(s)
        mu = torch.clamp(mu, self.action_low, self.action_high)
        sigma = self.fcsigma(s)
        sigma = F.softplus(sigma)
        
        m = normal.Normal(loc = mu, scale=sigma)

        return m
        

        