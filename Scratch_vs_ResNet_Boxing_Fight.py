from AtariScratch.boxing_double_dqn_agent import DQNAgent
from AtariPretrained.boxing_pretrained_dqn_agent import PretrainedDQNAgent
from AtariScratch.preprocessing import preprocess_env
from pettingzoo.atari import boxing_v2
from collections import Counter

ROUNDS = 11

env = boxing_v2.parallel_env() #render_mode="human"
env = preprocess_env(env)


scratch_agent = DQNAgent(num_channels=4, num_actions=18)
scratch_agent.load("boxing_dqn_scratch", 700)

resnet_agent = PretrainedDQNAgent(num_channels=4, num_actions=18)
resnet_agent.load("boxing_dqn_pretrained", 700)

scratch_agent_name = None
resnet_agent_name = None
scratch_wins, resnet_wins, draws = 0, 0, 0
scratch_action_counts = Counter()
resnet_action_counts = Counter()


for episode in range(ROUNDS):
    observations, infos = env.reset()

    if scratch_agent_name is None:
        scratch_agent_name = env.agents[0]
        resnet_agent_name = env.agents[1]

    scratch_reward = 0
    resnet_reward = 0
    for step in range(2000):
        scratch_action = scratch_agent.get_action(observations[scratch_agent_name], 0.1)
        resnet_action = resnet_agent.get_action(observations[resnet_agent_name], 0.1)

        scratch_action_counts[scratch_action]+=1
        resnet_action_counts[resnet_action]+=1

        actions={
            scratch_agent_name: scratch_action,
            resnet_agent_name: resnet_action
        }

        observations, rewards, terminations, truncations, infos = env.step(actions)
        scratch_reward += rewards[scratch_agent_name]
        resnet_reward += rewards[resnet_agent_name]

        if all(terminations.values()) or all(truncations.values()):
            break
    
    if scratch_reward > resnet_reward:
        scratch_wins += 1
    elif resnet_reward > scratch_reward:
        resnet_wins += 1
    else:
        draws += 1
    
    print(f"For episode {episode}: scratch got {scratch_reward} rewards, resnet got {resnet_reward} rewards")

print(f"Full results after {ROUNDS} episodes")
print(f"scratch wins: {scratch_wins}, resnet wins: {resnet_wins}, draws: {draws}")