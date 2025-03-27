from stable_baselines3 import DQN
from game_env import PygameAIEnv

env = PygameAIEnv()

model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, exploration_fraction=0.2, exploration_final_eps=0.1, 
            buffer_size=30000, batch_size=64, train_freq=4)

print("Bắt đầu huấn luyện...")
model.learn(total_timesteps=30000)

model.save("dqn_pygame_ai")
print("Huấn luyện hoàn tất! Mô hình đã được lưu.")
