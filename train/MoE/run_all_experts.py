# 🚀 One-Click Start: Sequentially train all expert models

# 1. Train Mild Expert
train_expert(
    expert_name="Mild",
    train_csv=r"D:\yifan_2025\data\moe_processed_data\expert_mild.csv",
    output_dir=r"D:\yifan_2025\data\moe_checkpoints\expert_mild",
    captions_csv=r"D:\yifan_2025\data\captions.csv"
)

print("\n" + "="*50)
print("✅ Mild Expert training completed, starting Moderate...")
print("="*50 + "\n")

# 2. Train Moderate Expert
train_expert(
    expert_name="Moderate",
    train_csv=r"D:\yifan_2025\data\moe_processed_data\expert_moderate.csv",
    output_dir=r"D:\yifan_2025\data\moe_checkpoints\expert_moderate",
    captions_csv=r"D:\yifan_2025\data\captions.csv"
)

print("\n" + "="*50)
print("✅ Moderate Expert training completed, starting Severe...")
print("="*50 + "\n")

# 3. Train Severe Expert
train_expert(
    expert_name="Severe",
    train_csv=r"D:\yifan_2025\data\moe_processed_data\expert_severe.csv",
    output_dir=r"D:\yifan_2025\data\moe_checkpoints\expert_severe",
    captions_csv=r"D:\yifan_2025\data\captions.csv"
)

print("\n" + "="*50)
print("🎉🎉🎉 All expert models have finished training!")
print("="*50 + "\n")
