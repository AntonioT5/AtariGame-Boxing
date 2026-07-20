from pettingzoo.atari import boxing_v2
from preprocessing import preprocess_env
from boxing_pretrained_dqn_agent import PretrainedDQNAgent
from collections import deque
import numpy as np

def quick_eval(agent, env, n_episodes=10):
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0
        for _ in range(2000):
            a = agent.get_action(obs["first_0"], 0)
            actions = {"first_0": a, "second_0": env.action_space("second_0").sample()}
            obs, rewards, term, trunc, _ = env.step(actions)
            ep_r += rewards["first_0"]
            if all(term.values()) or all(trunc.values()):
                break
        if ep_r > 0:
            wins += 1
    return wins / n_episodes


NUM_EPISODES = 701
MAX_STEPS_PER_EPISODE = 2000
TARGET_UPDATE_EVERY_STEPS = 2000
TRAIN_EVERY_N_STEPS = 16 #8
MIN_REPLAY_SIZE = 10000

env = boxing_v2.parallel_env()
env = preprocess_env(env)

agent = PretrainedDQNAgent(num_channels=4, num_actions=18)

epsilon = 1.0
epsilon_min = 0.2
epsilon_decay = 0.999997 #0.998

episode_rewards = []
recent_rewards = deque(maxlen=30)

steps_since_target_sync = 0

for episode in range(NUM_EPISODES):
    observation, infos = env.reset()
    episode_reward = {agent_name: 0 for agent_name in env.agents}

    for step in range(MAX_STEPS_PER_EPISODE):
        actions = {}

        for agent_name in env.agents:
            if agent_name == "first_0":
                actions[agent_name] = agent.get_action(observation[agent_name], epsilon)
            else:
                actions[agent_name] = env.action_space(agent_name).sample()

        next_observation, rewards, terminations, truncations, infos = env.step(actions)

        if "first_0" in env.agents:
            raw_reward = rewards["first_0"]
            scaled_reward = raw_reward * 0.1

            actual_done = terminations["first_0"]

            agent.update_memory(
                observation["first_0"],
                actions["first_0"],
                scaled_reward,
                next_observation["first_0"],
                actual_done
            )

        for agent_name in env.agents:
            episode_reward[agent_name] += rewards[agent_name]

        observation = next_observation
        steps_since_target_sync += 1

        if len(agent.memory) > MIN_REPLAY_SIZE:
            
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if step % TRAIN_EVERY_N_STEPS == 0:
                agent.train()

            if steps_since_target_sync >= TARGET_UPDATE_EVERY_STEPS:
                agent.update_target_model()
                steps_since_target_sync = 0

        if all(terminations.values()) or all(truncations.values()):
            break

    r0 = episode_reward["first_0"]
    r1 = episode_reward["second_0"]
    episode_rewards.append((r0, r1))
    recent_rewards.append(r0)
    rolling_avg = sum(recent_rewards) / len(recent_rewards)

    print(f"Episode {episode}: first_0 = {r0}, second_0 = {r1}, rolling_avg(30) = {rolling_avg:.2f} epsilon = {epsilon:.3f}")

    if episode % 50 == 0 and episode > 0:
        win_rate = quick_eval(agent, env)
        print(f"  >> Win rate vs random at episode {episode}: {win_rate:.2f}")

    if episode % 50 == 0:
        agent.save("boxing_dqn_pretrained", episode)

env.close()