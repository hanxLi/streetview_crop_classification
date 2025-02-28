import pandas as pd
import torch
import csv
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Evaluator:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class))

    def overall_accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def classwise_overal_accuracy(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return np.nan_to_num(acc)  # Replace NaNs with 0

    def precision(self):
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = tp / (tp + fp)
        return np.nan_to_num(precision)

    def recall(self):
        tp = np.diag(self.confusion_matrix)
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            recall = tp / (tp + fn)
        return np.nan_to_num(recall)

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
        return np.nan_to_num(f1)

    def intersection_over_union(self):
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = tp / (tp + fp + fn)
        return np.nan_to_num(iou)

    def _generate_matrix(self, ref_img, pred_img):
        mask = (ref_img >= 0) & (ref_img < self.num_class)
        label = self.num_class * ref_img[mask].astype(int) + pred_img[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, ref_img, pred_img):
        assert ref_img.shape == pred_img.shape, "Shape mismatch between reference and prediction."
        batch_size = ref_img.shape[0]
        for i in range(batch_size):
            self.confusion_matrix += self._generate_matrix(ref_img[i], pred_img[i])

    def plot_confusion_matrix(self, class_mapping, save_path="confusion_matrix.png"):
        row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
        conf_mat_normalized = np.divide(self.confusion_matrix, row_sums, where=row_sums != 0)

        classes = [class_mapping.get(i, f"Class {i}") for i in range(self.num_class)]
        df_cm = pd.DataFrame(conf_mat_normalized, index=classes, columns=classes)

        plt.figure(figsize=(self.num_class, self.num_class))
        sns.heatmap(df_cm, annot=True, fmt=".3f", cmap='viridis', linewidths=0.5, cbar=True)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Reference Label')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class))

def do_accuracy_evaluation(model, valData, num_classes, class_mapping, 
                           out_name=None, log_uncertainty=False):
    """
    Evaluate the model and compute metrics. Supports both models with and without uncertainty.
    """
    evaluator = Evaluator(num_classes)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    uncertainties = []

    with torch.no_grad():
        for _, batch in enumerate(valData):
            # Handle ancillary data and device transfer
            if len(batch) == 3:
                images, ancillary_data, labels = batch
                ancillary_data = ancillary_data.to(device)
            elif len(batch) == 2:
                images, labels = batch
                ancillary_data = None  # No ancillary data
            else:
                raise ValueError("Batch should contain 2 or 3 elements.")

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: Handle both models (with and without uncertainty)
            if ancillary_data is not None:
                output = model(images, ancillary_data)  # With ancillary data
            else:
                output = model(images)  # Without ancillary data

            # Handle model output (with or without uncertainty)
            if isinstance(output, tuple):
                logits, log_var = output  # Model with uncertainty
                if log_uncertainty:
                    uncertainty = torch.mean(torch.exp(log_var))
                    uncertainties.append(uncertainty.item())
            else:
                logits = output  # Model without uncertainty

            if torch.isnan(logits).any():
                print("Warning: NaN detected in model outputs.")

            # Compute predictions
            probabilities = F.softmax(logits, dim=1)
            _, preds = torch.max(probabilities, dim=1)

            # Add results to evaluator
            evaluator.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

    # Compute aggregated metrics
    metrics = {
        "Overall Accuracy": evaluator.overall_accuracy(),
        "Mean Accuracy": np.nanmean(evaluator.classwise_overal_accuracy()),
        "Mean IoU": np.nanmean(evaluator.intersection_over_union()),
        "Mean Precision": np.nanmean(evaluator.precision()),
        "Mean Recall": np.nanmean(evaluator.recall()),
        "Mean F1 Score": np.nanmean(evaluator.f1_score())
    }

    # Compute class-wise metrics
    classwise_metrics = {}
    for i, class_name in class_mapping.items():
        classwise_metrics[class_name] = {
            "Accuracy": evaluator.classwise_overal_accuracy()[i],
            "Precision": evaluator.precision()[i],
            "Recall": evaluator.recall()[i],
            "IoU": evaluator.intersection_over_union()[i],
            "F1 Score": evaluator.f1_score()[i]
        }

    # Log uncertainty if available
    if log_uncertainty and uncertainties:
        avg_uncertainty = np.mean(uncertainties)
        metrics["Mean Uncertainty"] = avg_uncertainty
        print(f"Mean Uncertainty: {avg_uncertainty:.4f}")

    # If out_name is provided, extract the directory path to save confusion matrix
    if out_name:
        save_dir = Path(out_name).parent  # Extract directory from out_name
        confusion_matrix_path = save_dir / "confusion_matrix.png"

        print(f"Saving confusion matrix to: {confusion_matrix_path}")

        # Plot and save the confusion matrix in the same folder as the CSV
        evaluator.plot_confusion_matrix(class_mapping, save_path=confusion_matrix_path)

        # Save metrics to CSV
        with open(out_name, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
            writer.writerow([])  # Add a blank line before class-wise metrics
            writer.writerow(["Class", "Accuracy", "Precision", "Recall", "IoU", "F1 Score"])
            for class_name, class_metrics in classwise_metrics.items():
                writer.writerow([class_name] + list(class_metrics.values()))

    return metrics, classwise_metrics
