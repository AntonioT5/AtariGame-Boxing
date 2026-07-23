from pettingzoo.atari import boxing_v2
from preprocessing import preprocess_env
from boxing_pretrained_dqn_agent import PretrainedDQNAgent
import numpy as np
import torch
from collections import Counter

env = boxing_v2.parallel_env(render_mode="human")
env = preprocess_env(env)


NUM_TRAINING_EPISODES = 1
agent = PretrainedDQNAgent(num_channels=4, num_actions=18)

agent.load("boxing_dqn_pretrained", 700)

trained_agent_name = None
random_agent_name = None
wins, losses, draws = 0,0,0
action_counts = Counter()

for episode in range(NUM_TRAINING_EPISODES):
    observations, infos = env.reset()

    if trained_agent_name is None:
        trained_agent_name = env.agents[0]
        random_agent_name = env.agents[1]

    ep_reward = 0
    for step in range(2000):
        trained_action = agent.get_action(observations[trained_agent_name], 0)
        action_counts[trained_action]+=1
        actions={
            trained_agent_name: trained_action,
            random_agent_name: env.action_space(random_agent_name).sample()
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        ep_reward += rewards[trained_agent_name]

        if all(terminations.values()) or all(truncations.values()):
            break
    
    if ep_reward>0:
        wins+=1
    elif ep_reward<0:
        losses+=1
    else:
        draws+=1
    
    print(f"For episode {episode} our agent got {ep_reward} rewards")
print(f"Full results after {NUM_TRAINING_EPISODES} episodes")
print(f"wins: {wins}, losses: {losses}, draws: {draws}")

print("Action distribution of trained agent during eval:")
for action, count in action_counts.most_common():
    print(f"Action {action}: chosen {count} times ({100*count/sum(action_counts.values()):.1f}%)")

