import matplotlib.pyplot as plt
import gymnasium as gym
from torch import nn

def plot_learning_process(scores: list[float], vals1: list[float], vals2: list[float]) -> None:
    '''
    Plots the training scores, losses, and epsilon values.

    Args:
    scores (List[float]): The training scores.
    losses (List[float]): The training losses.
    epsilons (List[float]): The epsilon values.

    Returns:
    None
    '''

    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    window = 50
    plt.plot(scores)
    if len(scores) >= window:
        running_avg = [sum(scores[max(0, i - window):i]) / window for i in range(len(scores))]
        plt.plot(running_avg, label=f'Running Avg (window={window})')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid(True)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(vals1)
    plt.xlabel("Update step")
    plt.grid(True)
    plt.subplot(133)
    plt.plot(vals2)
    plt.xlabel("Update step")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def validate(env: gym.Env, model: nn.Module):
    '''
    validate the model on the environment

    Args:
    - env: The environment to validate the model on.
    - model: The model to validate.
    '''
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = model(state).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state

    return total_reward
