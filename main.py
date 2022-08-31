import gym

env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode="human")

# Example of a game play
env.action_space.seed(42)
observation = env.reset(seed=42)
print("Initial observation: ", observation)
for _ in range(5):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("****")
    print("Applied action: ", action)
    print("Current observation: ", observation)
    print("Current reward: ", reward)
    print("Done? ", done)
    if done:
        observation = env.reset()
        print("New observation: ", observation)
env.close()
print("-----")
print("End of testing gameplay")