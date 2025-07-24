import torch
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
from implementations.utils import plot_learning_process


def compute_returns(rewards, dones, next_value, gamma, device):
    """
    Optimized returns computation using pre-allocated tensor
    """
    n_steps = len(rewards)
    returns = torch.zeros(n_steps, dtype=torch.float, device=device)

    R = next_value
    for i in reversed(range(n_steps)):
        R = rewards[i] + gamma * R * (1 - float(dones[i]))
        returns[i] = R

    return returns


def rollout(env, policy, critic, n_steps, device):
    """
    Optimized rollout with pre-allocated tensors and efficient tensor operations
    """
    log_probs = torch.zeros(n_steps, device=device)
    values = torch.zeros(n_steps, device=device)
    rewards = torch.zeros(n_steps, device=device)
    entropies = torch.zeros(n_steps, device=device)
    dones = torch.zeros(n_steps, dtype=torch.bool, device=device)

    state, _ = env.reset()
    done = False
    actual_steps = 0

    for step in range(n_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)

        probs = policy(state_tensor)
        value = critic(state_tensor).squeeze()

        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs[step] = log_prob
        values[step] = value
        rewards[step] = reward
        entropies[step] = entropy
        dones[step] = done

        state = next_state
        actual_steps += 1

        if done:
            break

    # Trim tensors to actual steps taken
    if actual_steps < n_steps:
        log_probs = log_probs[:actual_steps]
        values = values[:actual_steps]
        rewards = rewards[:actual_steps]
        entropies = entropies[:actual_steps]
        dones = dones[:actual_steps]

    if done:
        next_value = torch.tensor(0.0, device=device)
    else:
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device)
        next_value = critic(next_state_tensor).squeeze()

    return log_probs, values, rewards, entropies, dones, next_value, rewards.sum().item()


def update_models(optimizer_policy, optimizer_critic, log_probs, values, returns, entropies, entropy_coeff):
    """
    Optimized model update - tensors are already properly formatted
    """
    advantages = returns - values

    policy_loss = -(log_probs * advantages.detach()).sum() - entropy_coeff * entropies.sum()
    critic_loss = F.mse_loss(values, returns)

    optimizer_policy.zero_grad()
    optimizer_critic.zero_grad()

    policy_loss.backward()
    critic_loss.backward()

    optimizer_policy.step()
    optimizer_critic.step()

    return policy_loss.item(), critic_loss.item()


def train_a2c(env, policy, critic, optimizer_policy, optimizer_critic,
              gamma: float, num_episodes: int, n_steps: int, entropy_coeff: float = 0.01, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = policy.to(device)
    critic = critic.to(device)

    episode_rewards = []
    actor_losses = []
    critic_losses = []

    pbar = tqdm(range(num_episodes), desc="Training A2C")
    for _ in pbar:
        log_probs, values, rewards, entropies, dones, next_value, total_reward = rollout(env, policy, critic, n_steps, device)
        returns = compute_returns(rewards, dones, next_value, gamma, device)
        policy_loss, critic_loss = update_models(optimizer_policy, optimizer_critic, log_probs, values, returns, entropies, entropy_coeff)

        episode_rewards.append(total_reward)
        actor_losses.append(policy_loss)
        critic_losses.append(critic_loss)

        if len(episode_rewards) >= 100:
            avg_reward = sum(episode_rewards[-100:]) / 100
        else:
            avg_reward = sum(episode_rewards) / len(episode_rewards)

        pbar.set_postfix({"avg_reward_100": f"{avg_reward:.2f}"})

    plot_learning_process(episode_rewards, actor_losses, critic_losses)

    return policy, critic


if __name__ == "__main__":
    import torch.nn as nn

    env = gym.make("CartPole-v1", max_episode_steps=1024)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    class PolicyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
                nn.Softmax(dim=-1)
            )

        def forward(self, x):
            return self.fc(x)

    class CriticNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet()
    critic = CriticNet()
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)

    train_a2c(env, policy, critic, optimizer_policy, optimizer_critic, 0.99, 1000, 1024, 0.01, device)
