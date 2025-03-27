import pandas as pd
from dowhy import CausalModel
import matplotlib.pyplot as plt

# 1️⃣ Đọc dữ liệu AI chơi game
df = pd.read_csv("game_data.csv")

# 2️⃣ Kiểm tra dữ liệu có đúng không
print(df.head())  # Xem trước 5 dòng đầu tiên
print(df.columns)  # Kiểm tra tên cột

# 3️⃣ Tạo một cột "Hành động phân loại"
df["Action_Up"] = (df["Action"] == 0).astype(int)  # Lên
df["Action_Down"] = (df["Action"] == 1).astype(int)  # Xuống
df["Action_Left"] = (df["Action"] == 2).astype(int)  # Trái
df["Action_Right"] = (df["Action"] == 3).astype(int)  # Phải

# 4️⃣ Danh sách các hành động cần phân tích
actions = ["Action_Up", "Action_Down", "Action_Left", "Action_Right"]

# 5️⃣ Duyệt qua từng hành động để phân tích
results = {}
for action in actions:
    print(f"\n🔍 Phân tích hành động: {action}")

    # 6️⃣ Xây dựng mô hình nhân quả cho từng hành động
    model = CausalModel(
        data=df,
        treatment=action,
        outcome="Reward",
        common_causes=["Distance", "Blocked"]
    )

    # 7️⃣ Xác định và ước lượng hiệu ứng nhân quả
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

    # 8️⃣ Kiểm định giả thuyết
    refutation = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")

    # 9️⃣ Lưu kết quả
    results[action] = {
        "Causal Estimate": estimate.value,
        "Refutation Test": refutation
    }

# 🔟 Hiển thị kết quả tổng quan
print("\n📊 Kết quả tổng quan về từng hành động:")
for action, result in results.items():
    print(f"{action}: Causal Estimate = {result['Causal Estimate']:.4f}")

# 11️⃣ Vẽ biểu đồ so sánh hiệu ứng nhân quả của từng hành động
plt.figure(figsize=(8, 6))
plt.bar(results.keys(), [res["Causal Estimate"] for res in results.values()], color=['blue', 'green', 'red', 'orange'])
plt.xlabel("Hành động")
plt.ylabel("Tác động nhân quả lên phần thưởng")
plt.title("So sánh hiệu ứng nhân quả của từng hành động AI")
plt.show()
