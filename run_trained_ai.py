import pandas as pd
from stable_baselines3 import DQN
from game_env import PygameAIEnv


env = PygameAIEnv()
model = DQN.load("dqn_pygame_ai")

obs = env.reset()
done = False
data = []

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    data.append([
        obs[2],  # Distance to Goal
        obs[3], # Blocked
        action, reward
    ])    

# Lưu vào CSV để dùng cho phân tích doWhy
df = pd.DataFrame(data, columns=[
    "Distance",
    "Blocked",
    "Action", "Reward"
])
df.to_csv("game_data.csv", index=False)

print("Đã thu thập dữ liệu AI chơi game và lưu vào game_data.csv")

env.close()
