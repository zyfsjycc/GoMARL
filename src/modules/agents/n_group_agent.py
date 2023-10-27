import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init

class GroupAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GroupAgent, self).__init__()
        self.args = args
        self.a_h_dim = args.rnn_hidden_dim
        self.action_dim = args.n_actions

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.hyper_group = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(args.hypernet_embed, args.hypernet_embed),
                                        nn.Tanh()
                                        )

        self.hyper_b = nn.Sequential(nn.Linear(args.hypernet_embed, args.n_actions))

        self.hyper_w = nn.Sequential(nn.Linear(args.hypernet_embed, args.rnn_hidden_dim * args.n_actions))        

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size() 
        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        h_detach = h.detach()
        group_state = self.hyper_group(h_detach)
        group_state_detach = group_state.detach()

        fc2_w = self.hyper_w(group_state_detach).reshape(b*a, self.a_h_dim, self.action_dim)
        fc2_b = self.hyper_b(group_state_detach).reshape(b*a, 1, self.action_dim)
        _h = h.reshape(b*a, 1, self.a_h_dim)
        q = th.matmul(_h, fc2_w) + fc2_b

        return q.view(b, a, -1), h.view(b, a, -1), group_state.view(b, a, -1)