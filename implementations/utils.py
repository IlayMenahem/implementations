import matplotlib.pyplot as plt
import gymnasium as gym
from torch import nn

def plot_learning_process(*lists):
    '''
    Plots the training process values.

    Args:
    - *lists: A variable number of lists containing values to plot.

    '''
    plt.figure(figsize=(20, 10))

    for i, vals in enumerate(lists):
        plt.plot(vals, label=f'Series {i+1}')

    plt.legend()
    plt.grid(True)
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
