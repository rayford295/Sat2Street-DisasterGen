# =================== evaluate_resnet_cas_metrics.py ===================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm.auto import tqdm

# ================= 🔧 Configuration =================
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")

# Root directory for generated evaluation results
EVAL_ROOT = r"D:\yifan_2025\evaluation_results"
OUTPUT_DIR = os.path.join(EVAL_ROOT, "resnet_cas_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of methods to evaluate
METHODS = [
    "Real_GroundTruth",  # baseline (Upper Bound)
    "Pix2Pix",
    "SD1.5_ControlNet",
    "SD1.5_ControlNet_Gemini",
    "MoE_Ours"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Label mapping (normalize different naming styles to 0, 1, 2)
LABEL_TO_INT = {
    "0": 0, "mild": 0, "0_MinorDamage": 0, "no-damage": 0,
    "1": 1, "moderate": 1, "1_ModerateDamage": 1, "minor-damage": 1, "major-damage": 1,
    "2": 2, "severe": 2, "2_SevereDamage": 2, "2_Destroyed": 2, "destroyed": 2, "2_MajorDamage": 2,
    "Mild": 0, "Moderate": 1, "Severe": 2
}
INT_TO_LABEL = {0: "Mild", 1: "Moderate", 2: "Severe"}

# ================= 🛠️ Dataset Definitions =================

# 1. Real training dataset for ResNet (from CSV)
class RealTrainDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # For classification, we use street-view images
        img_name = os.path.basename(str(row['svi_path']))
        img_path = os.path.join(self.img_root, img_name)
        
        # Label processing
        raw_label = str(row['severity'])
        label = LABEL_TO_INT.get(raw_label, 1)  # default to Moderate
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            # Error tolerance: if image cannot be read, create a black image
            image = torch.zeros((3, 224, 224))
            
        return image, label

# 2. Generated dataset for evaluation (from folder structure)
class GeneratedDataset(Dataset):
    def __init__(self, method_name, root_dir, transform=None):
        """
        Directory structure: root_dir/Method/Mild/xxx.png
        """
        self.samples = []
        self.transform = transform
        
        if method_name == "Real_GroundTruth":
            # Special case: real test set from CSV
            df = pd.read_csv(TEST_CSV)
            for _, row in df.iterrows():
                name = os.path.basename(str(row['svi_path']))
                path = os.path.join(IMAGE_DIR, name)
                raw_l = str(row['severity'])
                label = LABEL_TO_INT.get(raw_l, 1)
                if os.path.exists(path):
                    self.samples.append((path, label))
        else:
            # Read generated results folder
            method_path = os.path.join(root_dir, method_name)
            for cat_name in ["Mild", "Moderate", "Severe"]:
                cat_dir = os.path.join(method_path, cat_name)
                if not os.path.exists(cat_dir): continue
                
                label = LABEL_TO_INT[cat_name]
                # We only read _gen.png or _fake.png files (filter out _real.png)
                files = os.listdir(cat_dir)
                for f in files:
                    if method_name != "Real_GroundTruth":
                        # Only evaluate generated images
                        if f.endswith("_gen.png") or (method_name == "Pix2Pix" and not f.endswith("_real.png")):
                            self.samples.append((os.path.join(cat_dir, f), label))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros((3, 224, 224))
        return image, label

# ================= 🧠 Train ResNet18 (Oracle) =================

def train_classifier():
    print("\n🔄 [Step 1] Training ResNet18 classifier (Oracle)...")
    save_path = os.path.join(OUTPUT_DIR, "resnet18_oracle.pth")
    
    if os.path.exists(save_path):
        print("✅ Found pre-trained classifier, loading directly.")
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(save_path))
        model.to(DEVICE)
        model.eval()
        return model

    # Data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = RealTrainDataset(TRAIN_CSV, IMAGE_DIR, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3-class classification
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train for 10 epochs
    model.train()
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)
            
    # Save
    torch.save(model.state_dict(), save_path)
    print(f"💾 Classifier saved at: {save_path}")
    model.eval()
    return model

# ================= 📊 Evaluation and Visualization =================

def evaluate_methods(model):
    print("\n🔄 [Step 2] Evaluating all generation methods...")
    
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = {}  # Store predictions and ground truth
    metrics_summary = []

    for method in METHODS:
        print(f"   Evaluating: {method}")
        dataset = GeneratedDataset(method, EVAL_ROOT, transform=eval_transforms)
        
        if len(dataset) == 0:
            print(f"   ⚠️ Warning: {method} dataset is empty, skipping.")
            continue
            
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"   Inferencing {method}", leave=False):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        results[method] = {
            "true": all_labels,
            "pred": all_preds,
            "acc": acc,
            "f1": f1
        }
        
        metrics_summary.append({
            "Method": method,
            "Accuracy": acc,
            "Macro F1": f1
        })
        
        print(f"   -> Acc: {acc:.4f} | F1: {f1:.4f}")

    # Save results
    df = pd.DataFrame(metrics_summary)
    df.to_csv(os.path.join(OUTPUT_DIR, "cas_metrics.csv"), index=False)
    print("\n✅ Evaluation completed, metrics saved.")
    return results, df

def plot_confusion_matrices(results):
    print("\n🔄 [Step 3] Plotting confusion matrices...")
    
    methods = list(results.keys())
    num_methods = len(methods)
    
    fig, axes = plt.subplots(1, num_methods, figsize=(4 * num_methods, 4))
    if num_methods == 1: axes = [axes]
    
    classes = ["Mild", "Mod", "Sev"]
    
    for ax, method in zip(axes, methods):
        data = results[method]
        cm = confusion_matrix(data['true'], data['pred'])
        
        # Normalize (percentage)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, ax=ax, cbar=False)
        
        ax.set_title(f"{method}\nAcc: {data['acc']:.2f} | F1: {data['f1']:.2f}")
        ax.set_xlabel("Predicted")
        if method == methods[0]:
            ax.set_ylabel("Ground Truth")
        else:
            ax.set_yticks([])
            
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "confusion_matrices_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Confusion matrices saved at: {save_path}")
    plt.show()

# ================= 🚀 Main =================

if __name__ == "__main__":
    # 1. Train or load classifier
    classifier = train_classifier()
    
    # 2. Evaluate all methods
    results, df = evaluate_methods(classifier)
    
    # 3. Plot results
    if results:
        plot_confusion_matrices(results)
        
    print("\n🏆 Final CAS Ranking:")
    print(df.sort_values("Macro F1", ascending=False).to_string(index=False))
