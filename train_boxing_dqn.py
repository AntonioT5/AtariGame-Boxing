from pettingzoo.atari import boxing_v2
from preprocessing import preprocess_env
from boxing_dqn_agent import DQNAgent

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 2000
TARGET_UPDATE_EVERY = 5
TRAIN_EVERY_N_STEPS = 4

env = boxing_v2.parallel_env()
env = preprocess_env(env)

agent = DQNAgent(num_channels=4, num_actions=18)

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

episode_rewards = []

for episode in range(NUM_EPISODES):
    observation, infos = env.reset()
    episode_reward = {agent_name: 0 for agent_name in env.agents}

    for step in range(MAX_STEPS_PER_EPISODE):
        actions = {}

        for agent_name in env.agents:
            actions[agent_name] = agent.get_action(observation[agent_name], epsilon)

        next_observation, rewards, terminations, truncations, infos = env.step(actions)

        for agent_name in env.agents:
            done = terminations[agent_name] or truncations[agent_name]
            agent.update_memory(observation[agent_name], actions[agent_name], rewards[agent_name], next_observation[agent_name], done)
            episode_reward[agent_name] += rewards[agent_name]

        observation = next_observation
        
        if step % TRAIN_EVERY_N_STEPS == 0:
            agent.train()

        if all(terminations.values()) or all(truncations.values()):
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if episode % TARGET_UPDATE_EVERY == 0:
        agent.update_target_model()

    total_reward = sum(episode_reward.values())
    episode_rewards.append(total_reward)
    print(f"Episode {episode}: total reward = {total_reward}, epssilon = {epsilon:.3f}")

    if episode % 50 == 0:
        agent.save("boxing_dqn_scratch", episode)

env.close()
