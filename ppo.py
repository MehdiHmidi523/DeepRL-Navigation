import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PPO():
    def __init__(
        self,
        model,
        batch_size=2000
    ):
        self.lr = [1e-4, 2e-4]
        self.gamma = 0.99
        self.batch_size = batch_size
        self.eps_clip = 0.2
        self.memory_counter = 0
        self.recollection = {"s": [], "a": [], "r": [], "sn": [], "end": [], "logp": [], "return": []}
        self.actor = model[0]()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        self.critic = model[1]()
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])

    def save_load_model(self, op, path):
        anet_path = path + "ppo_anet.pt"
        cnet_path = path + "ppo_cnet.pt"
        if op == "save":
            torch.save(self.actor.state_dict(), anet_path)
            torch.save(self.critic.state_dict(), cnet_path)
        elif op == "load":
            self.actor.load_state_dict(torch.load(anet_path))
            self.critic.load_state_dict(torch.load(cnet_path))

    def choose_action(self, s, eval=False):
        s_ts = torch.FloatTensor(np.expand_dims(s, 0))
        if not eval:
            a_ts, policy_step = self.actor.sample(s_ts)
            a_ts = torch.clamp(a_ts, min=-1, max=1)
            action = a_ts.cpu().detach().numpy()[0]
            return action, policy_step.cpu().detach().numpy()[0]
        else:
            a_ts, policy_step = self.actor.sample(s_ts)
            a_ts = torch.clamp(a_ts, min=-1, max=1)
            return  a_ts.cpu().detach().numpy()[0]

    def store_transition(self, s, a, r, sn, end, logp):
        self.recollection["s"].append(s)
        self.recollection["a"].append(a)
        self.recollection["r"].append(r)
        self.recollection["sn"].append(sn)
        self.recollection["end"].append(end)
        self.recollection["logp"].append(logp)
        self.memory_counter += 1

    def run_return(self):
        self.recollection["return"] = []
        discounted_reward = 0
        for reward, end in zip(reversed(self.recollection["r"]), reversed(self.recollection["end"])):
            if end == 0:
                discounted_reward = reward
            discounted_reward = reward + (self.gamma * discounted_reward)
            self.recollection["return"].insert(0, discounted_reward)

    def learn(self):
        self.run_return()

        s_ts = torch.FloatTensor(np.array(self.recollection["s"]))
        a_ts = torch.FloatTensor(np.array(self.recollection["a"]))
        r_ts = torch.FloatTensor(np.expand_dims(np.array(self.recollection["r"]), 1))
        sn_ts = torch.FloatTensor(np.array(self.recollection["sn"]))
        end_ts = torch.FloatTensor(np.expand_dims(np.array(self.recollection["end"]), 1))

        logp_ts = torch.FloatTensor(np.expand_dims(np.array(self.recollection["logp"]), 1))
        calculated_return = torch.FloatTensor(np.expand_dims(np.array(self.recollection["return"]), 1))
        calculated_return = (calculated_return - calculated_return.mean()) / (calculated_return.std() + 1e-5)

        for it in range(1):
            dist = self.actor.distribution(s_ts)
            policy_selection = dist.log_prob(a_ts)
            ent = dist.entropy()
            value = self.critic(s_ts)

            ratio = (policy_selection - logp_ts.detach()).exp()
            advantage = calculated_return - value.detach()
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            
            gradient_loss = (-advantage * policy_selection).mean()
            val_loss = torch.nn.MSELoss()(value, calculated_return).mean()
            entropy_loss = ent.mean()

            loss = gradient_loss + 0.5 * val_loss - 0.01 * entropy_loss

            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()

            loss.backward()

            self.critic_optim.step()
            self.actor_optim.step()
            if it % 10 == 0:
                print("iteration", it, \
                      ", gradient_loss:", gradient_loss.detach().cpu().numpy(), \
                      ", entropy_loss:", entropy_loss.detach().cpu().numpy(), \
                      ", val_loss:", val_loss.detach().cpu().numpy())
