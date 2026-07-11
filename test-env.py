from pettingzoo.atari import boxing_v2

env = boxing_v2.parallel_env()
observations, infos = env.reset()

nonzero_events = 0

for step in range(2000):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)

    for agent, r in rewards.items():
        if r != 0:
            nonzero_events += 1
            print(f"Step {step}: {agent} got reward {r}")

    if all(terminations.values()) or all(truncations.values()):
        print(f"Episode ended at step {step}")
        break

print(f"Total nonzero reward events in this episode: {nonzero_events}")
env.close()