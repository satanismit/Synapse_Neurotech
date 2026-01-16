import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import os
from config import Config
from dataset import SynapseDataset
from model import EMGConv1D

def evaluate_and_save():
    # Ensure output directory exists
    Config.ensure_dirs()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load Data
    val_dataset = SynapseDataset(Config.DATA_ROOT, subjects=Config.VAL_SUBJECTS, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load Model
    model = EMGConv1D().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Model not found! Please train first.")
        return

    model.eval()
    all_preds = []
    all_labels = []

    # Inference Loop
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    
    # Format Output
    output_text = []
    output_text.append("="*50)
    output_text.append("   MODEL EVALUATION REPORT")
    output_text.append("="*50)
    output_text.append(f"Model Path: {Config.MODEL_PATH}")
    output_text.append(f"Validation Subjects: {Config.VAL_SUBJECTS}")
    output_text.append("-" * 50)
    output_text.append(f"Accuracy: {acc:.4f}")
    output_text.append(f"Macro F1 Score: {f1:.4f}")
    output_text.append("-" * 50)
    output_text.append("\nClassification Report:\n")
    output_text.append(report)
    output_text.append("-" * 50)
    output_text.append("\nConfusion Matrix:\n")
    output_text.append(str(cm))
    output_text.append("\n" + "="*50)

    # Save to File
    output_path = os.path.join(Config.OUTPUT_DIR, "evaluation_results.txt")
    output_path = os.path.abspath(output_path)
    
    with open(output_path, "w") as f:
        f.write("\n".join(output_text))
    
    print("\n".join(output_text))
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    evaluate_and_save()
