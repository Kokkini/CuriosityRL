import torch
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
obs_size = env.observation_space.shape[0]
num_acts = env.action_space.n

eta = 10


# logit net
layers = []
sizes = [obs_size, 64, 64, num_acts]
for i in range(len(sizes)-2):
    layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
    layers.append(torch.nn.ReLU())
layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
logit_net = torch.nn.Sequential(*layers)
print(logit_net)



# next state prediction net
layers = []
sizes = [obs_size+num_acts, 64, 64, obs_size]
for i in range(len(sizes) - 2):
    layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
    layers.append(torch.nn.ReLU())
layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
state_net = torch.nn.Sequential(*layers)
print(state_net)

vis_dict = {"reward": [],
            "loss_pi": [],
            "loss_state": []}

def to_state_net_input(act_arr, state_arr, num_acts):
    act_onehot = np.zeros((len(act_arr), num_acts))
    act_onehot[np.arange(len(act_arr)), act_arr] = 1
    result = np.concatenate((act_onehot, state_arr), axis=1)
    return result


def plot_dict(d, sharex=True, start=0):
    fig, axs = plt.subplots(len(d), 1, sharex=sharex, figsize=(10, len(d)*5))
    keys = list(d.keys())
    for i in range(len(keys)):
        ax = axs[i]
        key = keys[i]
        ax.plot(d[key][start:])
        ax.set_title(key)
    plt.show()

def get_policy(obs):
    logits = logit_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()


def reward_to_go(rews, gamma):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + gamma*(rtgs[i+1] if i+1 < n else 0)
    return list(rtgs)

def n_step_reward(rews, vals, gamma, n):
    result = []
    rew_to_go = reward_to_go(rews, gamma)
    T = len(rews) - 1
    for t in range(T+1):
        if t+n > T:
            r = rew_to_go[t]
        else:
            r = rew_to_go[t] - (gamma**n)*rew_to_go[t+n] + (gamma**n)*vals[t+n]
        result.append(r)
    return result

def n_step_reward_at(t, n, rew_to_go, vals, gamma):
    T = len(rew_to_go) - 1
    if t + n > T:
        r = rew_to_go[t]
    else:
        r = rew_to_go[t] - (gamma ** n) * rew_to_go[t + n] + (gamma ** n) * vals[t + n]
    return r

def GAE_reward(rews, vals, gamma, lam):
    rew_to_go = reward_to_go(rews, gamma)
    T = len(rews) - 1
    result = []
    for t in range(len(rews)):
        if t < T:
            r = 0
            for n in range(1, T-t):
                r += n_step_reward_at(t, n, rew_to_go, vals, gamma) * lam**(n-1)
            r *= (1 - lam)
            r += (lam**(T-t-1)) * n_step_reward_at(t, T-t, rew_to_go, vals, gamma)
        else:
            r = n_step_reward_at(t, 1, rew_to_go, vals, gamma)
        result.append(r)
    return result

def train_one_epoch(policy_optimizer, state_optimizer, batch_size, target_kl=0.01, render = True):
    obs = env.reset()
    batch_obs = []
    batch_obs_after = []
    batch_act = []
    ep_act = []
    batch_weight = []
    batch_reward_to_go = []
    ep_val = []
    ep_reward = []
    ep_obs = []
    ep_obs_after = []
    batch_log_prob = []
    render_this_epoch = False
    gamma = 0.99
    batch_ep_reward = []
    batch_all_reward = []

    while True:
        if not render_this_epoch and render:
            env.render()
        batch_obs.append(obs.copy())
        ep_obs.append(obs.copy())
        act_dist = get_policy(torch.as_tensor(obs, dtype=torch.float32))
        act = act_dist.sample().item()
        batch_log_prob.append(act_dist.log_prob(torch.as_tensor(act)).item())
        batch_act.append(act)
        ep_act.append(act)
        obs, reward, done, info = env.step(act)
        batch_obs_after.append(obs.copy())
        ep_obs_after.append(obs.copy())
        ep_reward.append(reward)

        if done:
            batch_ep_reward.append(sum(ep_reward))
            render_this_epoch = True
            # weights = list(reward_to_go(ep_reward, gamma))
            # weights = n_step_reward(ep_reward, ep_val, gamma, 30)
            state_net_input = to_state_net_input(np.array(ep_act), np.array(ep_obs), num_acts)
            state_pred = state_net(torch.as_tensor(state_net_input, dtype=torch.float32))

            intrinsic_rewards = eta * np.mean((state_pred.data.numpy() - np.array(ep_obs_after)) ** 2, axis=1)
            # ep_reward = np.array(ep_reward) + intrinsic_rewards
            ep_reward = intrinsic_rewards
            batch_all_reward.append(sum(ep_reward))

            # weights = GAE_reward(ep_reward, ep_val, gamma, lam=0.97)
            # batch_weight += weights
            batch_reward_to_go += reward_to_go(ep_reward, gamma)
            obs = env.reset()
            ep_reward = []
            ep_val = []
            ep_obs = []
            ep_act = []
            ep_obs_after = []
            done = False
            if len(batch_obs) >= batch_size:
                break


    for i in range(1):
        policy_optimizer.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                                  torch.as_tensor(batch_act, dtype=torch.long),
                                  torch.as_tensor(batch_reward_to_go, dtype=torch.float32))


        batch_loss.backward()
        policy_optimizer.step()

    for k in range(1):
        state_optimizer.zero_grad()
        state_net_input = to_state_net_input(np.array(batch_act), np.array(batch_obs), num_acts)
        state_pred = state_net(torch.as_tensor(state_net_input, dtype=torch.float32))

        state_loss = torch.mean((state_pred - torch.as_tensor(batch_obs_after, dtype=torch.float32)) ** 2)
        state_loss.backward()
        state_optimizer.step()


    print(f"loss: {batch_loss.item()}")
    print(f"state loss: {state_loss.item()}")
    avg_ep_reward = sum(batch_ep_reward)/len(batch_ep_reward)
    print(f"ep reward: {avg_ep_reward}")
    print(f"ep all reward: {sum(batch_all_reward)/len(batch_all_reward)}")

    vis_dict["reward"].append(avg_ep_reward)
    vis_dict["loss_pi"].append(batch_loss.item())
    vis_dict["loss_state"].append(state_loss.item())



batch_size = 1000
num_epochs = 500
policy_optimizer = torch.optim.Adam(logit_net.parameters(), lr=1e-2)
state_optimizer = torch.optim.Adam(state_net.parameters(), lr=1e-2)


for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    train_one_epoch(policy_optimizer, state_optimizer, batch_size, target_kl=0.01, render=True)

plot_dict(vis_dict, start=10)

