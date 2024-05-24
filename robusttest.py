import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
import os
import argparse
import torch
import dgl
from dgl.nn import GATConv
import numpy as np
import matplotlib.pyplot as plt

# Created directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# GNN to encode body graph structure
class AntGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_size, num_heads=1)
        self.conv2 = GATConv(hidden_size, hidden_size, num_heads=1)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = torch.relu(x)
        x = self.conv2(g, x)
        return x

# Custom SB3 policy
class CustomAntPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn = AntGNN(2, 64)  # Adjusted the input features based on your node features

    def forward(self, obs, deterministic=False):
        # Converted the observation to a graph
        graph = self.obs_to_graph(obs)
        # Get graph embeddings
        x = self.gnn(graph, graph.ndata['feat'])
        # Flattened the embeddings
        x = x.flatten().unsqueeze(0)  # Add a batch dimension
        # Extracted features using the features extractor
        features = self.extract_features(obs, self.observation_space.shape)
        # Concatenated the features with the graph embeddings
        x = torch.cat([features, x], dim=1)
        # Passed through actor and critic networks
        return self.actor_critic(x)

    def obs_to_graph(self, obs):
        # Defining the edges between the nodes
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                                   [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]], dtype=torch.long)

        # Extracted the relevant features from the observation
        joint_angles = obs[:8]
        joint_velocities = obs[8:16]
        x_position = obs[16:17]
        y_position = obs[17:18]
        contact_forces = obs[18:26]
        z_position = obs[26:27]

        # Created node features
        node_features = torch.cat([joint_angles, joint_velocities, x_position, y_position, contact_forces, z_position], dim=0)

        # Created the DGL graph
        g = dgl.graph((edge_index[0], edge_index[1]))
        g.ndata['feat'] = node_features.unsqueeze(0)

        return g

def train(env, sb3_algo):
    match sb3_algo:
        case 'SAC':
            model = SAC('MultiInputPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir,
                        policy_kwargs={'net_arch': [64, 64], 'features_extractor_class': FlattenExtractor})
        case 'PPO':
            model = PPO('MultiInputPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir,
                        policy_kwargs={'net_arch': [64, 64], 'features_extractor_class': FlattenExtractor})
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 25000
    max_total_timesteps = 1000000
    total_timesteps = 0

    results = {
        'algorithm': sb3_algo,
        'timesteps': [],
        'rewards': [],
        'episode_lengths': []
    }

    while total_timesteps < max_total_timesteps:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        total_timesteps += TIMESTEPS
        timesteps = model.num_timesteps
        rewards = model.ep_info_buffer[-1]['r']
        episode_lengths = model.ep_info_buffer[-1]['l']
        results['timesteps'].append(timesteps)
        results['rewards'].append(rewards)
        results['episode_lengths'].append(episode_lengths)
        model.save(f"{model_dir}/{sb3_algo}_{total_timesteps}")

    return results

def plot_results(results_list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for results in results_list:
        algo = results['algorithm']
        timesteps = results['timesteps']
        rewards = results['rewards']
        episode_lengths = results['episode_lengths']

        ax1.plot(timesteps, rewards, label=algo)
        ax2.plot(timesteps, episode_lengths, label=algo)

    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()

    ax2.set_xlabel('Timesteps')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Episode Length')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def add_sensor_noise(obs, noise_std=0.1):
    noisy_obs = obs.copy()
    noise = np.random.normal(0, noise_std, size=obs.shape)
    noisy_obs[0:8] += noise[0:8]  # Added noise to joint angles
    noisy_obs[8:16] += noise[8:16]  # Added noise to joint velocities
    noisy_obs[18:26] += noise[18:26]  # Added noise to contact forces
    return noisy_obs

def test(env, sb3_algo, path_to_model, noise_std=0.1):
    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    print("Starting episode")
    episode_rewards = []

    while True:
        noisy_obs = add_sensor_noise(obs, noise_std)
        action, _ = model.predict(noisy_obs)
        obs, reward, done, _, _ = env.step(action)
        episode_rewards.append(reward)
        env.render()  # Rendered the environment for visualization
        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break

    print(f"Episode rewards: {episode_rewards}")
    print(f"Episode reward for {sb3_algo} with noise std {noise_std}: {sum(episode_rewards)}", flush=True)

def main(args):
    if args.train:
        gymenv = gym.make(args.gymenv, render_mode=None)
        results_list = []

        for algo in ['SAC', 'PPO']:
            results = train(gymenv, algo)
            results_list.append(results)

        plot_results(results_list)

    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')  
            noise_levels = [0.0, 0.1, 0.2, 0.3]
            for noise_std in noise_levels:
                test(gymenv, args.sb3_algo, path_to_model=args.test, noise_std=noise_std)
        else:
            print(f'{args.test} not found.')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', nargs='?', default='Ant-v4', help='Gymnasium environment (default: Ant-v4)')
    parser.add_argument('sb3_algo', nargs='?', default='SAC', help='StableBaseline3 RL algorithm (default: SAC)')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()
    main(args)