import pandas as pd
from dowhy import CausalModel
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file game_data.csv
df = pd.read_csv("game_data.csv")
print("Dữ liệu đầu vào:")
print(df.head())

# Bước 1: Xây dựng mô hình nhân quả
# Định nghĩa các biến:
# - Action: Hành động của AI (0: lên, 1: xuống, 2: trái, 3: phải)
# - Blocked: Trạng thái bị chặn (0: không, 1: có)
# - Distance: Khoảng cách tới mục tiêu
# - Reward: Phần thưởng nhận được
# Giả định: Action ảnh hưởng đến Blocked và Distance, từ đó ảnh hưởng đến Reward

# Tạo biểu đồ nhân quả (Causal Graph) dưới dạng GML
causal_graph = """
digraph {
    Action -> Blocked;
    Action -> Distance;
    Blocked -> Reward;
    Distance -> Reward;
}
"""

# Khởi tạo mô hình nhân quả với doWhy
model = CausalModel(
    data=df,
    treatment="Action",  # Biến điều trị (Treatment) - Hành động của AI
    outcome="Reward",   # Biến kết quả (Outcome) - Phần thưởng
    graph=causal_graph
)

# Hiển thị biểu đồ nhân quả
model.view_model()

# Bước 2: Xác định hiệu ứng nhân quả (Identify the causal effect)
identified_estimand = model.identify_effect()
print("Estimand được xác định:")
print(identified_estimand)

# Bước 3: Ước lượng hiệu ứng nhân quả (Estimate the causal effect)
# Sử dụng phương pháp hồi quy tuyến tính để ước lượng
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    target_units="ate"  # Average Treatment Effect
)

print("Ước lượng hiệu ứng nhân quả:")
print(causal_estimate)

# Bước 4: Kiểm tra độ bền của kết quả (Refutation)
# Sử dụng phương pháp thêm nhiễu ngẫu nhiên để kiểm tra tính mạnh mẽ của mô hình
refute_results = model.refute_estimate(
    identified_estimand,
    causal_estimate,
    method_name="random_common_cause"
)

print("Kết quả kiểm tra độ bền:")
print(refute_results)

# Bước 5: Phân tích và trực quan hóa kết quả
# Vẽ biểu đồ thể hiện mối quan hệ giữa Action và Reward
plt.figure(figsize=(10, 6))
for action in df["Action"].unique():
    subset = df[df["Action"] == action]
    plt.scatter(subset["Distance"], subset["Reward"], label=f"Action {action}", alpha=0.5)
plt.xlabel("Distance to Goal")
plt.ylabel("Reward")
plt.title("Reward vs Distance for Different Actions")
plt.legend()
plt.grid(True)
plt.show()

# Phân tích tác động của Blocked lên Reward
blocked_effect = df.groupby("Blocked")["Reward"].mean()
print("Tác động trung bình của Blocked lên Reward:")
print(blocked_effect)

# Trực quan hóa tác động của Blocked
plt.figure(figsize=(8, 5))
blocked_effect.plot(kind="bar", color=["blue", "red"])
plt.title("Average Reward by Blocked Status")
plt.xlabel("Blocked (0: No, 1: Yes)")
plt.ylabel("Average Reward")
plt.show()

blocked_counts = df.groupby(['Action', 'Blocked']).size().unstack(fill_value=0)

# Đặt tên cột cho rõ ràng (Blocked = 0: Không bị chặn, Blocked = 1: Bị chặn)
blocked_counts.columns = ['Not Blocked', 'Blocked']

# In kết quả thống kê để kiểm tra
print("Số lần Blocked cho mỗi Action:")
print(blocked_counts)

# Vẽ biểu đồ cột
blocked_counts.plot(kind='bar', stacked=False, figsize=(10, 6), color=['blue', 'red'])

# Tùy chỉnh biểu đồ
plt.title("Số lần Blocked theo mỗi Action", fontsize=14)
plt.xlabel("Action (0: Lên, 1: Xuống, 2: Trái, 3: Phải)", fontsize=12)
plt.ylabel("Số lần xuất hiện", fontsize=12)
plt.legend(title="Trạng thái Blocked")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=0)  # Giữ nhãn Action thẳng đứng

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()

print("Phân tích XAI hoàn tất!")