import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DDPG:
    def __init__(
            self,
            base_net,
            b_size=64
    ):
        self.tau = 0.005
        self.memory_size = 10000
        self.batch_size = b_size
        self.criterion = nn.MSELoss()
        self.eps_params = [1.0, 0.5, 0.00001]
        self._e = self.eps_params[0]
        self.recollection = {"s": [], "a": [], "r": [], "sn": [], "end": []}
        self.counter = 0
        self.lr = [0.0001, 0.0001]
        self.gamma = 0.99
        self.actor = base_net[0]()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic = base_net[1]()
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        self.critic_target = base_net[1]()
        self.critic_target.eval()

    def save_load_model(self, order, save_pos):
        anet_path = save_pos + "A_DDPG.pt"
        cnet_path = save_pos + "C_DDPG.pt"
        if order == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif order == "load":
            self.critic.load_state_dict(torch.load(cnet_path))
            self.critic_target.load_state_dict(torch.load(cnet_path))
            self.actor.load_state_dict(torch.load(anet_path))

    def choose_action(self, s, eval=False):
        A_ = self.actor(torch.FloatTensor(np.expand_dims(s, 0)))
        A_ = A_.cpu().detach().numpy()[0]
        if not eval:
            A_ += np.random.normal(0, self._e, A_.shape)
        else:
            A_ += np.random.normal(0, self.eps_params[1], A_.shape)
        return np.clip(A_, -1, 1)

    def store_transition(self, s, a, r, sn, end):
        if self.counter <= self.memory_size:
            self.recollection["s"].append(s)
            self.recollection["a"].append(a)
            self.recollection["r"].append(r)
            self.recollection["sn"].append(sn)
            self.recollection["end"].append(end)
        else:
            mem = self.counter % self.memory_size
            self.recollection["s"][mem] = s
            self.recollection["a"][mem] = a
            self.recollection["r"][mem] = r
            self.recollection["sn"][mem] = sn
            self.recollection["end"][mem] = end

        self.counter += 1

    def softie(self):
        with torch.no_grad():
            for target_p, eval_p in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_p.copy_((1 - self.tau) * target_p.data + self.tau * eval_p.data)

    def learn(self):
        if self.counter < self.memory_size:
            sample_index = np.random.choice(self.counter, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)

        s_batch = [self.recollection["s"][m] for m in sample_index]
        a_batch = [self.recollection["a"][m] for m in sample_index]
        r_batch = [self.recollection["r"][m] for m in sample_index]
        sn_batch = [self.recollection["sn"][m] for m in sample_index]
        end_batch = [self.recollection["end"][m] for m in sample_index]

        s_ts = torch.FloatTensor(np.array(s_batch))
        a_ts = torch.FloatTensor(np.array(a_batch))
        r_ts = torch.FloatTensor(np.array(r_batch))
        sn_ts = torch.FloatTensor(np.array(sn_batch))
        end_ts = torch.FloatTensor(np.array(end_batch))

        with torch.no_grad():
            next_action = self.actor(sn_ts)
            q_target = r_ts + end_ts * self.gamma * self.critic_target(sn_ts, next_action)

        self.c_loss = self.criterion(self.critic(s_ts, a_ts), q_target)

        self.critic_optim.zero_grad()
        self.c_loss.backward()
        self.critic_optim.step()

        val_current = self.critic(s_ts, self.actor(s_ts))
        self.a_loss = -val_current.mean()

        self.actor_optim.zero_grad()
        self.a_loss.backward()
        self.actor_optim.step()

        self.softie()

        if self._e > self.eps_params[1]:
            self._e -= self.eps_params[2]
        else:
            self._e = self.eps_params[1]

        return float(self.a_loss.detach().cpu().numpy()), float(self.c_loss.detach().cpu().numpy())
