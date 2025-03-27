import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from game_env import PygameAIEnv

# Load game và mô hình AI đã huấn luyện
env = PygameAIEnv()
model = DQN.load("dqn_pygame_ai")

data = []

obs = env.reset()
done = False

for _ in range(30000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    # Lưu dữ liệu với thông tin mở rộng
    data.append([
        obs[0], obs[1], obs[2],  # Player X, Player Y, Distance to Goal
        obs[3], # Blocked
        action, reward
    ])    

    if done == True:
        obs = env.reset()
        done = False

# Lưu vào CSV để dùng cho phân tích doWhy
df = pd.DataFrame(data, columns=[
    "Player X", "Player Y", "Distance",
    "Blocked",
    "Action", "Reward"
])
df.to_csv("game_data.csv", index=False)

print("Đã thu thập dữ liệu AI chơi game và lưu vào game_data.csv")
