import gym
import pygame
import numpy as np
from gym import spaces

WIDTH, HEIGHT = 600, 400
PLAYER_SIZE = 20
OBSTACLE_SIZE = 40
OBSTACLE_NUM = 6

class PygameAIEnv(gym.Env):
    def __init__(self):
        super(PygameAIEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # 4 hành động: lên, xuống, trái, phải
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]),  
                                            high=np.array([WIDTH, HEIGHT, np.sqrt(WIDTH**2 + HEIGHT**2), 1]),  
                                            dtype=np.float32)
        
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.reset()

    def distance(self, x1, y1, x2, y2):
        """ Tính khoảng cách Euclidean giữa hai điểm """
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def reset(self):
        self.player_x, self.player_y = (40, 40)#np.random.randint(50, WIDTH-50), np.random.randint(50, HEIGHT-50)
        self.goal_x, self.goal_y = (WIDTH - 60, HEIGHT - 60)#np.random.randint(50, WIDTH-50), np.random.randint(50, HEIGHT-50)
        self.obstacles = []
        for _ in range(OBSTACLE_NUM):  # Số lượng vật cản
            while True:
                obs_x, obs_y = (np.random.randint(50, WIDTH-50), np.random.randint(50, HEIGHT-50))
                if (self.distance(obs_x, obs_y, self.player_x, self.player_y) > 40 and
                        self.distance(obs_x, obs_y, self.goal_x, self.goal_y) > 40):
                    self.obstacles.append((obs_x, obs_y))
                    break  # Thoát vòng lặp khi tìm được vị trí hợp lệ

        self.blocked = False
        return self._get_state()

    def _get_state(self):
        distance_to_goal = self.distance(self.player_x, self.player_y,self.goal_x, self.goal_y)

        return np.array([self.player_x, self.player_y, distance_to_goal, int(self.blocked)], 
        dtype=np.float32)

    def step(self, action):
        dx, dy = 0, 0
        if action == 0:   dy = -5  # Lên
        elif action == 1: dy = 5   # Xuống
        elif action == 2: dx = -5  # Trái
        elif action == 3: dx = 5   # Phải

         # Kiểm tra va chạm với vật cản
        for obs in self.obstacles:
            if pygame.Rect(self.player_x + dx, self.player_y + dy, PLAYER_SIZE, PLAYER_SIZE).colliderect(
                                       pygame.Rect(obs[0], obs[1], OBSTACLE_SIZE, OBSTACLE_SIZE)):
                dx, dy = 0, 0  # Nếu có va chạm thì dừng lại

        # Cập nhật vị trí mới
        
        new_x = np.clip(self.player_x + dx, 0, WIDTH - 20)  # Giới hạn trong [0, WIDTH-20]
        new_y = np.clip(self.player_y + dy, 0, HEIGHT - 20) # Giới hạn trong [0, HEIGHT-20]
        distance_to_goal = self.distance(self.player_x, self.player_y,self.goal_x, self.goal_y)

        reward = -distance_to_goal

        if new_x == self.player_x and new_y == self.player_y:   #Kiểm tra xem có bị va chạm không
            self.blocked = True
            reward -= 20
        else:
            self.blocked = False
            reward += 10


        done = False
        if pygame.Rect(self.player_x + dx, self.player_y + dy, PLAYER_SIZE, PLAYER_SIZE).colliderect(
                                       pygame.Rect(self.goal_x, self.goal_y, 40, 40)):
            done = True
  

        self.player_x = new_x
        self.player_y = new_y
        return self._get_state(), reward, done, {}

    def render(self):
        self.screen.fill((255, 255, 255))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.player_x, self.player_y, PLAYER_SIZE, PLAYER_SIZE))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.goal_x, self.goal_y, 40, 40))
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (0, 0, 0), (obs[0], obs[1], OBSTACLE_SIZE, OBSTACLE_SIZE))
        pygame.display.update()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
