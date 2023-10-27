import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from utils.th_utils import get_parameters_num

class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.abs = abs
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.hypernet_dim = args.hypernet_embed
        self.grouping_hypernet_dim = args.grouping_hypernet_embed
        
        self.group = args.group
        if self.group is None:
            self.group = [[_ for _ in range(self.n_agents)]]

        self.group_turn_now = 0

        self.a_h_dim = args.rnn_hidden_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.hyper_w1 = nn.ModuleList(nn.Sequential(nn.Linear(self.a_h_dim, self.grouping_hypernet_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.grouping_hypernet_dim, self.embed_dim))
                                        for i in range(self.n_agents))

        self.hyper_b1 = nn.ModuleList([nn.Sequential(nn.Linear(self.a_h_dim, self.embed_dim))])

        self.hyper_w2 = nn.ModuleList([nn.Sequential(nn.Linear(self.hypernet_dim, self.hypernet_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hypernet_dim, self.embed_dim))])
        
        self.hyper_b2 = nn.ModuleList([nn.Sequential(nn.Linear(self.hypernet_dim, self.embed_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.embed_dim, 1))])

        self.hyper_w3 = nn.Sequential(nn.Linear(self.hypernet_dim, self.hypernet_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hypernet_dim, self.embed_dim))
        self.hyper_b3 = nn.Sequential(nn.Linear(self.hypernet_dim, self.embed_dim))

        self.hyper_w4 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hypernet_dim, self.embed_dim))
        self.hyper_b4 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.embed_dim, 1))

    def add_new_net(self):
        self.hyper_b1.append(nn.Sequential(nn.Linear(self.a_h_dim, self.embed_dim)))

        self.hyper_w2.append(nn.Sequential(nn.Linear(self.hypernet_dim, self.hypernet_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.hypernet_dim, self.embed_dim)))
        
        self.hyper_b2.append(nn.Sequential(nn.Linear(self.hypernet_dim, self.embed_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(self.embed_dim, 1)))
        if self.args.use_cuda:
            self.hyper_b1.cuda()
            self.hyper_w2.cuda()
            self.hyper_b2.cuda()
        
        print("New Mixer Size:")
        print(get_parameters_num(self.parameters()))
    
    def del_net(self, idx):
        del self.hyper_b1[idx]
        del self.hyper_w2[idx]
        del self.hyper_b2[idx]

        print("Del Group {} Network Mixer Size:".format(idx))
        print(get_parameters_num(self.parameters()))

    def forward(self, qvals, states, a_h, all_group_state, which_network):
        b, t, _ = qvals.size()
        states = states.reshape(b*t, -1)
        group_q_list = []
        group_state_list = []
        w1_list = []
        w1_avg_list = []
        similarity_loss_list = []

        a_h = a_h.reshape(b * t, self.n_agents, -1)
        all_group_state = all_group_state.reshape(b * t, self.n_agents, -1)

        for group_index, group_i in enumerate(self.group):
            group_state = all_group_state[:, group_i, :]
            group_state = th.max(group_state, dim=1)[0]
            group_state_list.append(group_state)


        for i in range(self.n_agents):
            w1 = self.hyper_w1[i](a_h[:, i, :])
            if self.abs:
                w1 = w1.abs()
            w1_list.append(w1)
        
        w1 = th.stack(w1_list, dim=1)
   
        sd_loss = 0
        for group_index, group_i in enumerate(self.group):
            group_n = len(group_i)
            group_qs = qvals[:, :, group_i]
            group_qs = group_qs.reshape(b * t, 1, group_n)

            group_a_h = a_h[:, group_i, :]

            group_w1 = w1[:, group_i, :]

            b1 = self.hyper_b1[group_index](group_a_h)
            b1 = th.sum(b1, dim=1, keepdim=True)
            
            group_j = []
            for group_tmp in self.group:
                if group_tmp != group_i:
                    group_j += group_tmp

            for agent_i in group_i:
                group_state_agent_i = all_group_state[:, agent_i, :].unsqueeze(1)
                group_state_sim = all_group_state[:, group_i, :]
                group_state_div = all_group_state[:, group_j, :]
                group_state_agent_i_sim = group_state_agent_i.expand_as(group_state_sim)
                group_state_agent_i_div = group_state_agent_i.expand_as(group_state_div)

                sim_loss = th.sum(group_state_agent_i_sim * group_state_sim, dim=-1) / \
                                ((th.sum(group_state_agent_i_sim ** 2, dim=-1) ** 0.5) * (th.sum(group_state_sim ** 2, dim=-1) ** 0.5))
                div_loss = th.sum(group_state_agent_i_div * group_state_div, dim=-1) / \
                                ((th.sum(group_state_agent_i_div ** 2, dim=-1) ** 0.5) * (th.sum(group_state_div ** 2, dim=-1) ** 0.5))

                sd_loss -= th.sum(sim_loss, dim=-1)
                sd_loss += th.sum(div_loss, dim=-1)

            group_state = group_state_list[group_index]
            w2 = self.hyper_w2[group_index](group_state).view(b*t, self.embed_dim, 1)
            b2 = self.hyper_b2[group_index](group_state).view(b*t, 1, 1)
            if self.abs:
                w2 = w2.abs()
            
            group_w1_avg = th.mean(th.mean(group_w1, dim=2, keepdim=False), dim=0, keepdim=False)
            if self.group_turn_now == group_index:
                w1_avg_list.append(group_w1_avg)

            hidden = F.elu(th.matmul(group_qs, group_w1) + b1)
            q_group = th.matmul(hidden, w2) + b2

            group_q_list.append(q_group.view(b*t, 1))

        group_tot_q = th.stack(group_q_list, dim=-1)
        group_state_stack = th.stack(group_state_list, dim=1)
        w3 = self.hyper_w3(group_state_stack)
        b3 = self.hyper_b3(group_state_stack)
        b3 = th.sum(b3, dim=1, keepdim=True)

        w4 = self.hyper_w4(states).view(b*t, self.embed_dim, 1)
        b4 = self.hyper_b4(states).view(b*t, 1, 1)

        if self.abs:
            w3 = w3.abs()
            w4 = w4.abs()
        
        tot_hidden = F.elu(th.matmul(group_tot_q, w3) + b3)
        tot_q = th.matmul(tot_hidden, w4) + b4
        
        return tot_q.view(b, t, -1) , w1_avg_list, sd_loss.view(b, t, -1) / (self.n_agents * self.n_agents)
    
    def get_w1_avg(self, a_h):
        b, t, _, _ = a_h.size()
        a_h = a_h.reshape(b * t, self.n_agents, -1)
        
        w1_list = []
        for i in range(self.n_agents):
            w1 = self.hyper_w1[i](a_h[:, i, :])
            if self.abs:
                w1 = w1.abs()
            w1_list.append(w1)
        
        w1 = th.stack(w1_list, dim=1)
        w1_avg = th.mean(th.mean(w1, dim=2, keepdim=False), dim=0, keepdim=False)
        return w1_avg
    
    def update_group(self, new_group):
        self.group = new_group
        self.group_turn_now = (self.group_turn_now + 1) % len(self.group)


