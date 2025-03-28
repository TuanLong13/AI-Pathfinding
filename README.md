# AI Pathfinding game với DQN và Phân tích nhân quả bằng DoWhy

Dự án này sử dụng mô hình Deep Q-Network (DQN) để huấn luyện AI chơi một trò chơi tìm đường đi trong môi trường Pygame, sau đó phân tích các hành động của AI bằng DoWhy để tìm hiểu mối quan hệ nhân quả trong mỗi hành động.

## 1. Cài đặt

### Yêu cầu Python
Dự án yêu cầu Python phiên bản **3.8 trở lên**. Bạn có thể kiểm tra phiên bản Python hiện tại bằng lệnh sau:

```bash
python --version
```

Nếu chưa cài đặt Python, bạn có thể tải xuống tại [python.org](https://www.python.org/downloads/).

### Cài đặt thư viện
Trước khi chạy dự án, bạn cần cài đặt các thư viện cần thiết. Sử dụng lệnh sau:

```bash
pip install stable-baselines3 gym pygame numpy pandas matplotlib dowhy
```

## 2. Các tệp trong dự án

- `game_env.py` - Định nghĩa môi trường trò chơi trong Pygame.
- `train_dqn.py` - Huấn luyện mô hình DQN cho AI.
- `run_trained_ai.py` - Chạy mô hình đã huấn luyện và thu thập dữ liệu chơi game.
- `collect_data.py` - Thu thập dữ liệu trong 30000 bước đi của mô hình đã huấn luyện
- `analyze_doWhy.py` - Phân tích dữ liệu bằng DoWhy để tìm hiểu nhân quả.
- `analyze_action_detail.py` - Phân tích chi tiết tác động của từng hành động AI.

## 3. Hướng dẫn sử dụng

### Bước 1: Huấn luyện AI
Chạy lệnh sau để huấn luyện mô hình AI:

```bash
python train_dqn.py
```

Sau khi huấn luyện xong, mô hình sẽ được lưu dưới tên `dqn_pygame_ai.zip`.

### Bước 2: Chạy AI đã huấn luyện
Sau khi có mô hình, bạn có thể chạy AI để quan sát và thu thập dữ liệu:

```bash
python run_trained_ai.py
```

Dữ liệu chơi game sẽ được lưu vào `game_data.csv`.

Ngoài ra có thể thu thập dữ liệu trong 30000 bước đi bằng cách chạy:

```bash
python collect_data.py
```
Dữ liệu vẫn sẽ được lưu vào `game_data.csv`.

### Bước 3: Phân tích nhân quả bằng DoWhy
Bạn có thể thực hiện phân tích nhân quả bằng cách chạy:

```bash
python analyze_doWhy.py
```

Điều này sẽ đưa ra các kết quả ước lượng nhân quả và tạo các biểu đồ phân tích quan hệ nhân quả giữa các biến đầu vào.

### Bước 4: Phân tích chi tiết hành động
Nếu muốn phân tích chi tiết hơn từng hành động AI, hãy chạy:

```bash
python analyze_action_detail.py
```
