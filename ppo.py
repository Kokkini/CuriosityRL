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

# value net
layers = []
sizes = [obs_size, 64, 64, 1]
for i in range(len(sizes) - 2):
    layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))
    layers.append(torch.nn.ReLU())
layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
value_net = torch.nn.Sequential(*layers)
print(value_net)


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
            "loss_v": [],
            "loss_state": []}

def to_state_net_input(act_arr, state_arr, num_acts):
    act_onehot = np.zeros((len(act_arr), num_acts))
    act_onehot[np.arange(len(act_arr)), act_arr] = 1
    result = np.concatenate((act_onehot, state_arr), axis=1)
    return result


def plot_dict(d, sharex=True):
    fig, axs = plt.subplots(len(d), 1, sharex=sharex, figsize=(10, len(d)*5))
    keys = list(d.keys())
    for i in range(len(keys)):
        ax = axs[i]
        key = keys[i]
        ax.plot(d[key])
        ax.set_title(key)
    plt.show()

def get_policy(obs):
    logits = logit_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, old_log_prob, advatage, epsilon):
    new_log_prob = get_policy(obs).log_prob(act)
    policy_ratio = torch.exp(new_log_prob - old_log_prob)
    clamped_policy_ratio = torch.clamp(policy_ratio, 1-epsilon, 1+epsilon)
    loss = - torch.min(policy_ratio*advatage, clamped_policy_ratio*advatage).mean()

    approx_kl = (old_log_prob - new_log_prob).mean().item()
    info_dict = dict(kl=approx_kl)
    return loss, info_dict



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

def train_one_epoch(policy_optimizer, value_optimizer, state_optimizer, batch_size, target_kl=0.01, render = True):
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
    epsilon = 0.2
    gamma = 0.99
    updates_per_epochs = 10
    batch_ep_reward = []
    batch_all_reward = []

    while True:
        if not render_this_epoch and render:
            env.render()
        batch_obs.append(obs.copy())
        ep_obs.append(obs.copy())
        act_dist = get_policy(torch.as_tensor(obs, dtype=torch.float32))
        val = value_net(torch.as_tensor(obs, dtype=torch.float32)).item()
        ep_val.append(val)
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

            weights = GAE_reward(ep_reward, ep_val, gamma, lam=0.97)
            batch_weight += weights
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



    advatages = torch.as_tensor(batch_reward_to_go, dtype=torch.float32) - value_net(torch.as_tensor(batch_obs, dtype=torch.float32))
    advatages = advatages.data.numpy()
    for j in range(updates_per_epochs):
        loss_advatages = value_net(torch.as_tensor(batch_obs, dtype=torch.float32)) - torch.as_tensor(batch_reward_to_go,dtype=torch.float32)
        value_optimizer.zero_grad()
        value_loss = (loss_advatages ** 2).mean()


        value_loss.backward()
        value_optimizer.step()
    print(f"value loss: {value_loss.item()}")

    for i in range(updates_per_epochs):
        policy_optimizer.zero_grad()
        batch_loss, pi_info = compute_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                                  torch.as_tensor(batch_act, dtype=torch.long),
                                  torch.as_tensor(batch_log_prob, dtype=torch.float32),
                                  torch.as_tensor(advatages, dtype=torch.float32),
                                  epsilon)

        kl = pi_info['kl']
        if kl > 1.5 * target_kl:
            print('Early stopping at step %d due to reaching max kl.' % i)
            break

        batch_loss.backward()
        policy_optimizer.step()

    for k in range(updates_per_epochs):
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
    vis_dict["loss_v"].append(value_loss.item())
    vis_dict["loss_state"].append(state_loss.item())



batch_size = 1000
num_epochs = 30
policy_optimizer = torch.optim.Adam(logit_net.parameters(), lr=1e-3)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
state_optimizer = torch.optim.Adam(state_net.parameters(), lr=1e-3)


for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    train_one_epoch(policy_optimizer, value_optimizer, state_optimizer, batch_size, target_kl=0.01, render=True)

plot_dict(vis_dict)

