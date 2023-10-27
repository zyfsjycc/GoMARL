import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.group import Mixer as GroupMixer
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from utils.th_utils import get_parameters_num


class GROUPLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.params = list(self.mac.parameters())

        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')

        if args.mixer == "group":
            self.mixer = GroupMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        
        self.train_t = 0

        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden = []
        mac_group_state = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_hidden.append(self.mac.hidden_states)
            mac_group_state.append(self.mac.group_states)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)
        mac_hidden = th.stack(mac_hidden, dim=1)
        mac_group_state = th.stack(mac_group_state, dim=1)
        mac_hidden = mac_hidden.detach()

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Mixer
        chosen_action_qvals, w1_avg_list, sd_loss = self.mixer(chosen_action_qvals, batch["state"][:, :-1], mac_hidden[:, :-1], mac_group_state[:, :-1], "eval")

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            target_mac_out = []
            target_mac_hidden = []
            target_mac_group_state = []

            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_hidden.append(self.target_mac.hidden_states)
                target_mac_group_state.append(self.target_mac.group_states)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_mac_hidden = th.stack(target_mac_hidden, dim=1)
            target_mac_group_state = th.stack(target_mac_group_state, dim=1)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"], target_mac_hidden, target_mac_group_state, "target")

            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                            self.args.n_agents, self.args.gamma, self.args.td_lambda)
        
        # lasso_alpha
        lasso_alpha = []
        for i in range(len(w1_avg_list)):
            lasso_alpha_time = self.args.lasso_alpha_start * (self.args.lasso_alpha_anneal ** (t_env//self.args.lasso_alpha_anneal_time))
            lasso_alpha.append(lasso_alpha_time)

        # lasso loss
        lasso_loss = 0
        for i in range(len(w1_avg_list)):
            group_w1_sum = th.sum(w1_avg_list[i])
            lasso_loss += group_w1_sum * lasso_alpha[i]
        
        sd_loss = sd_loss * mask
        sd_loss = self.args.sd_alpha * sd_loss.sum() / mask.sum()

        td_error = (chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = masked_td_error.sum() / mask.sum()

        loss = td_loss + lasso_loss + sd_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self.logger.log_stat("total_loss", loss.item(), t_env)
            self.logger.log_stat("lasso_loss", lasso_loss.item(), t_env)
            self.logger.log_stat("sd_loss", sd_loss.item(), t_env)
            
            self.log_stats_t = t_env
    
    def change_group(self, batch: EpisodeBatch, change_group_i: int):
        if change_group_i == 0:
            self.agent_w1_avg = 0

        mac_hidden = []

        with th.no_grad():
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_hidden.append(self.mac.hidden_states)
            mac_hidden = th.stack(mac_hidden, dim=1)
            
            w1_avg = self.mixer.get_w1_avg(mac_hidden[:, :-1])
            self.agent_w1_avg += w1_avg

        if change_group_i == self.args.change_group_batch_num - 1:
            
            self.agent_w1_avg /= self.args.change_group_batch_num
            group_now = copy.deepcopy(self.mixer.group)
            group_nxt = copy.deepcopy(self.mixer.group)
            for group_index, group_i in enumerate(group_now):
                group_w1_avg = self.agent_w1_avg[group_i]

                group_avg = th.mean(group_w1_avg)
                relative_lasso_threshold = group_avg * self.args.change_group_value
                indices = th.where(group_w1_avg < relative_lasso_threshold)[0]

                if len(group_i) < 3:
                    continue
                
                if group_index+1 == len(group_now) and len(indices) != 0:
                    tmp = []
                    group_nxt.append(tmp)
                    self.mixer.add_new_net()
                    self.target_mixer.add_new_net()
                
                for i in range(len(indices)-1, -1, -1):
                    idx = group_now[group_index][indices[i]]
                    group_nxt[group_index+1].append(idx)
                    del group_nxt[group_index][indices[i]]
                    for m in self.mixer.hyper_w1[idx]:
                        if type(m) != nn.ReLU:
                            m.reset_parameters()
            
            whether_group_changed = True if group_now != group_nxt else False
            
            if not whether_group_changed:
                return
            
            for i in range(len(group_nxt)-1, -1, -1):
                if group_nxt[i] == []:
                    del group_nxt[i]
                    self.mixer.del_net(i)
                    self.target_mixer.del_net(i)
            
            self.mixer.update_group(group_nxt)
            self.target_mixer.update_group(group_nxt)
            self._update_targets()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            np.save("{}/group.npy".format(path), self.mixer.group)
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.group = np.load("{}/group.npy".format(path))
            for i in range(len(self.mixer.group)-1):
                self.mixer.add_new_net()
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
