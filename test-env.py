from pettingzoo.atari import boxing_v2
import time

env = boxing_v2.parallel_env(render_mode="human")

observations, info = env.reset()

for step in range(200):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observation, reward, terminations, truncations, infos = env.step(actions)

    for agent, reward in reward.items():
        if reward!=0:
            print(f"Step {step}: {agent} got reward {reward}")
    
    if all(terminations.values()) or all(truncations.values()):
        print("Episode finished, resetting...")
        observation, info = env.reset()

    time.sleep(0.2)

env.close()
