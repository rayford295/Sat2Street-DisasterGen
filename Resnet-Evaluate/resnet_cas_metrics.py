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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm.auto import tqdm

# ================= 🔧 Configuration =================
BASE_DIR = r"D:\yifan_2025\data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "train_split.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")

# Root directory for results
EVAL_ROOT = r"D:\yifan_2025\evaluation_results"
OUTPUT_DIR = os.path.join(EVAL_ROOT, "resnet_cas_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of methods to evaluate
METHODS = [
    "Real_GroundTruth", 
    "Pix2Pix",
    "SD1.5_ControlNet",
    "SD1.5_ControlNet_Gemini",
    "MoE"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_TO_INT = {
    "0": 0, "mild": 0, "0_MinorDamage": 0, "no-damage": 0,
    "1": 1, "moderate": 1, "1_ModerateDamage": 1, "minor-damage": 1, "major-damage": 1,
    "2": 2, "severe": 2, "2_SevereDamage": 2, "2_Destroyed": 2, "destroyed": 2, "2_MajorDamage": 2,
    "Mild": 0, "Moderate": 1, "Severe": 2
}
INT_TO_LABEL = {0: "Mild", 1: "Moderate", 2: "Severe"}

# ================= 🛠️ Dataset Definitions =================

class RealTrainDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = os.path.basename(str(row['svi_path']))
        img_path = os.path.join(self.img_root, img_name)
        raw_label = str(row['severity'])
        label = LABEL_TO_INT.get(raw_label, 1)
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros((3, 224, 224))
            
        return image, label, img_name

class GeneratedDataset(Dataset):
    def __init__(self, method_name, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        
        folder_name = "MoE_Ours" if method_name == "MoE" else method_name

        if method_name == "Real_GroundTruth":
            df = pd.read_csv(TEST_CSV)
            for _, row in df.iterrows():
                name = os.path.basename(str(row['svi_path']))
                path = os.path.join(IMAGE_DIR, name)
                raw_l = str(row['severity'])
                label = LABEL_TO_INT.get(raw_l, 1)
                if os.path.exists(path):
                    self.samples.append((path, label, name))
        else:
            method_path = os.path.join(root_dir, folder_name)
            for cat_name in ["Mild", "Moderate", "Severe"]:
                cat_dir = os.path.join(method_path, cat_name)
                if not os.path.exists(cat_dir): continue
                
                label = LABEL_TO_INT[cat_name]
                files = os.listdir(cat_dir)
                for f in files:
                    if method_name != "Real_GroundTruth":
                        if f.endswith("_gen.png") or (method_name=="Pix2Pix" and not f.endswith("_real.png")):
                            self.samples.append((os.path.join(cat_dir, f), label, f))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, name = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros((3, 224, 224))
        return image, label, name

# ================= 🧠 Train ResNet18 (Oracle) =================

def train_classifier():
    print("\n🔄 [Step 1] Training/Loading ResNet18 Classifier...")
    save_path = os.path.join(OUTPUT_DIR, "resnet18_oracle.pth")
    
    if os.path.exists(save_path):
        print("✅ Found pretrained classifier, loading...")
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(save_path))
        model.to(DEVICE)
        model.eval()
        return model

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = RealTrainDataset(TRAIN_CSV, IMAGE_DIR, transform=train_transforms)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    model.train()
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        
        for inputs, labels, _ in pbar:
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
            
    torch.save(model.state_dict(), save_path)
    print(f"💾 Classifier saved to: {save_path}")
    model.eval()
    return model

# ================= 📊 Evaluation & Plotting =================

def evaluate_methods(model):
    print("\n🔄 [Step 2] Evaluating all generation methods...")
    
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = {} 
    metrics_summary = []
    detailed_predictions = []
    cm_data_list = [] # To store raw confusion matrix data

    for method in METHODS:
        print(f"   Evaluating: {method}")
        dataset = GeneratedDataset(method, EVAL_ROOT, transform=eval_transforms)
        
        if len(dataset) == 0:
            print(f"   ⚠️ Warning: Dataset for {method} is empty, skipping.")
            continue
            
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_filenames = []
        
        with torch.no_grad():
            for inputs, labels, filenames in tqdm(loader, desc=f"   Inferencing {method}", leave=False):
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_filenames.extend(filenames)
        
        # --- 1. Calculate Standard Metrics ---
        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=["Mild", "Moderate", "Severe"], output_dict=True)
        
        # Macro Avg Metrics
        macro_prec = report['macro avg']['precision']
        macro_rec = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        
        # --- 2. Calculate Confusion Matrix & Per-Class Accuracy ---
        # Confusion Matrix: Row = True, Col = Pred
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        
        # Calculate Per-Class Accuracy (Recall)
        # Avoid division by zero
        class_counts = cm.sum(axis=1)
        per_class_acc = np.divide(cm.diagonal(), class_counts, out=np.zeros_like(cm.diagonal(), dtype=float), where=class_counts!=0)
        
        mild_acc = per_class_acc[0]
        mod_acc  = per_class_acc[1]
        sev_acc  = per_class_acc[2]
        
        # --- 3. Organize Summary Data ---
        summary_row = {
            "Method": method,
            "Accuracy": acc,          
            "Precision": macro_prec,  
            "Recall": macro_rec,      
            "F1_Score": macro_f1,     
            # 🔥 NEW: Per-Class Accuracy
            "Mild_Acc": mild_acc,
            "Mod_Acc": mod_acc,
            "Sev_Acc": sev_acc
        }
        metrics_summary.append(summary_row)
        
        # --- 4. Organize Raw Confusion Matrix Data ---
        # Saving raw counts: Mild_True_Mild_Pred, Mild_True_Mod_Pred, etc.
        cm_row = {
            "Method": method,
            # True Mild
            "TrueMild_PredMild": cm[0,0], "TrueMild_PredMod": cm[0,1], "TrueMild_PredSev": cm[0,2],
            # True Moderate
            "TrueMod_PredMild": cm[1,0], "TrueMod_PredMod": cm[1,1], "TrueMod_PredSev": cm[1,2],
            # True Severe
            "TrueSev_PredMild": cm[2,0], "TrueSev_PredMod": cm[2,1], "TrueSev_PredSev": cm[2,2],
        }
        cm_data_list.append(cm_row)

        # --- 5. Organize Per-Image Predictions ---
        for fname, true_l, pred_l in zip(all_filenames, all_labels, all_preds):
            detailed_predictions.append({
                "Method": method,
                "Filename": fname,
                "GroundTruth_Label": INT_TO_LABEL[true_l],
                "Predicted_Label": INT_TO_LABEL[pred_l],
                "Is_Correct": (true_l == pred_l)
            })
        
        results[method] = {
            "true": all_labels,
            "pred": all_preds,
            "acc": acc,
            "f1": macro_f1
        }
        
        print(f"   -> Acc: {acc:.4f} | F1: {macro_f1:.4f}")

    # --- Save Files ---
    
    # 1. Summary Metrics
    df_metrics = pd.DataFrame(metrics_summary)
    cols = ["Method", "Accuracy", "Recall", "Precision", "F1_Score", "Mild_Acc", "Mod_Acc", "Sev_Acc"]
    cols = [c for c in cols if c in df_metrics.columns] 
    df_metrics = df_metrics[cols]
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, "cas_metrics_summary.csv"), index=False)
    
    # 2. Raw Confusion Matrices
    df_cm = pd.DataFrame(cm_data_list)
    df_cm.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrices_raw.csv"), index=False)

    # 3. Per-image Predictions
    df_preds = pd.DataFrame(detailed_predictions)
    df_preds.to_csv(os.path.join(OUTPUT_DIR, "predictions_per_image.csv"), index=False)
    
    print("\n✅ Evaluation Completed!")
    print(f"   📄 Summary Metrics:    {os.path.join(OUTPUT_DIR, 'cas_metrics_summary.csv')}")
    print(f"   📄 Raw Conf Matrices:  {os.path.join(OUTPUT_DIR, 'confusion_matrices_raw.csv')}")
    print(f"   📄 Per-Image Preds:    {os.path.join(OUTPUT_DIR, 'predictions_per_image.csv')}")
    
    return results, df_metrics

def plot_confusion_matrices(results):
    print("\n🔄 [Step 3] Plotting Confusion Matrices...")
    
    methods = list(results.keys())
    num_methods = len(methods)
    
    fig, axes = plt.subplots(1, num_methods, figsize=(4 * num_methods, 4))
    if num_methods == 1: axes = [axes]
    
    classes = ["Mild", "Mod", "Sev"]
    
    for ax, method in zip(axes, methods):
        data = results[method]
        cm = confusion_matrix(data['true'], data['pred'])
        
        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
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
    print(f"🖼️ Confusion Matrices Visualization Saved: {save_path}")
    plt.show()

# ================= 🚀 Main Execution =================

if __name__ == "__main__":
    # 1. Train Classifier
    classifier = train_classifier()
    
    # 2. Evaluate
    results, df = evaluate_methods(classifier)
    
    # 3. Plot
    if results:
        plot_confusion_matrices(results)
        
    print("\n🏆 Final CAS Ranking (Sorted by F1_Score):")
    if not df.empty:
        # Displaying the expanded metrics list
        print(df.sort_values("F1_Score", ascending=False).to_string(index=False))
