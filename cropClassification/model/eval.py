import pandas as pd
import torch
import csv
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator(object):
    
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


    def overall_accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc


    def classwise_overal_accuracy(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc


    def precision(self):
        """
        Also known as User’s Accuracy (UA) and Positive Prediction Value (PPV).
        """
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        precision = np.where((tp + fp) != 0,
                             tp / (tp + fp),
                             0)
        return precision


    def recall(self):
        """
        Also known as Producer's Accuracy (PA), True Positive Rate, Sensitivity 
        and hit rate.
        """
        tp = np.diag(self.confusion_matrix)
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        recall = np.where((tp + fn) != 0,
                          tp / (tp + fn),
                          np.nan)
        return recall
    
    def f1_score(self):
        """
        Also known as balanced F-score or F-measure.

        The F1 score can be interpreted as a weighted average of the precision and 
        recall, where an F1 score reaches its best value at 1 and worst score at 0.
        """
        precision = self.precision()
        recall = self.recall()
        f1_score = np.where((precision + recall) != 0, 
                        2 * (precision * recall) / (precision + recall), 
                        0)
        return f1_score


    def intersection_over_union(self):
        tp = np.diag(self.confusion_matrix)
        # Predicted Positive (TP+FP); each column's sum
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        # Actual Positive (TP+FN); each row's sum
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        iou = np.where((tp + fp + fn) != 0,
                       tp / (tp + fp + fn),
                       0)
        return iou


    def _generate_matrix(self, ref_img, pred_img):
        """
        Generate confusion matrix for a given pair of ground truth and predicted
        images within a batch.

        For each pair in the batch, the resulting confusion matrix is a 2D array 
        where each row corresponds to a class in the ground truth, and each column 
        corresponds to a class in the prediction. The (i, j) element of the matrix 
        is the number of pixels that belong to class i in the ground truth and are 
        classified as class j in the prediction.

        Args:
            ref_img (np.array): 2D array of ref annotation.
            pred_img (np.array): 2D array of model's prediction.

        Returns:
            np.array: A 2D confusion matrix of size (num_class x num_class). 
                      Rows correspond to the true classes and columns correspond 
                      to the predicted classes.
        """
        mask = (ref_img >= 0) & (ref_img < self.num_class)
        label = self.num_class * ref_img[mask].astype('int') + pred_img[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix


    def add_batch(self, ref_img, pred_img):
        """
        update the cumulative confusion matrix with the results from a 
        new batch of images.
        """
        assert ref_img.shape == pred_img.shape
        batch_size = ref_img.shape[0]
        for i in range(batch_size):
            self.confusion_matrix += self._generate_matrix(ref_img[i], pred_img[i])

    def plot_confusion_matrix(self, class_mapping, save_path="confusion_matrix.png"):
        """
        Plot the confusion matrix and save it as an image file. Includes the background class (class 0).
        
        Args:
            class_mapping (dict): Dictionary mapping class indices to class names.
            save_path (str, optional): Path to save the confusion matrix image. Defaults to "confusion_matrix.png".
        """
        # Normalize the confusion matrix by row (i.e., by the true class)
        row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
        conf_mat_normalized = np.divide(self.confusion_matrix, row_sums, where=row_sums != 0)

        # Class labels for plotting, including background (class 0)
        classes = [class_mapping[i] for i in range(self.num_class)]

        # Create a dataframe for the seaborn heatmap
        df_cm = pd.DataFrame(conf_mat_normalized,
                             index=classes, 
                             columns=classes)
    
        # Create the figure
        plt.figure(figsize=(self.num_class, self.num_class))
    
        # Use seaborn to plot the heatmap
        heatmap = sns.heatmap(df_cm, annot=True, fmt=".3f", cmap='viridis', linewidths=.5, cbar=True)
    
        # Set the title and labels
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted label')
        plt.ylabel('Reference label')
    
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()


    def reset(self):
        """
        Resets the confusion matrix.

        This function sets the confusion matrix back to an empty state, ready to 
        start a new round of evaluation. It can be useful in cases where evaluation 
        is done in an episodic manner, such as when evaluating model performance after
        each epoch during training.
        """
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

#######################

def do_accuracy_evaluation(model, dataloader, num_classes, class_mapping, out_name=None):
    """
    Evaluate the performance of a trained model on a dataset with or without ancillary data 
    and calculate various metrics.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The dataloader for the evaluation dataset.
        num_classes (int): The number of target classes in the dataset.
        class_mapping (dict): A dictionary mapping class indices to class names.
        out_name (str, optional): The path where the evaluation metrics are to be saved. 
                                  If None, metrics are not saved. Defaults to None.

    Returns:
        dict: A dictionary containing the calculated metrics including Overall Accuracy, 
              Mean Accuracy, Mean IoU (Intersection over Union), mean Precision, and mean Recall.
    """
    evaluator = Evaluator(num_classes)
    model.eval()

    # Support for CUDA, MPS, and CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for data in dataloader:
            # Dynamically check if ancillary data is provided
            if len(data) == 3:
                images, labels, ancillary_data = data
                ancillary_data = ancillary_data.to(device)
                outputs = model(images.to(device), ancillary_data)
            elif len(data) == 2:
                images, labels = data
                outputs = model(images.to(device))
            else:
                raise ValueError("Dataloader must return either 2 or 3 elements (images, labels, [ancillary_data]).")

            # Handle NaN outputs
            if torch.isnan(outputs).any():
                print("NaN value found in model outputs!")

            # Calculate softmax probabilities and predictions
            outputs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)

            # Add the batch to the evaluator
            evaluator.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

    # Calculate evaluation metrics
    overall_accuracy = evaluator.overall_accuracy()
    classwise_overal_accuracy = evaluator.classwise_overal_accuracy()
    mean_accuracy = np.nanmean(classwise_overal_accuracy)
    IoU = evaluator.intersection_over_union()
    mean_IoU = np.nanmean(IoU)
    precision = evaluator.precision()
    mean_precision = np.nanmean(precision)
    recall = evaluator.recall()
    mean_recall = np.nanmean(recall)
    f1_score = evaluator.f1_score()
    mean_f1_score = np.nanmean(f1_score)

    metrics = {
        "Overall Accuracy": overall_accuracy,
        "Mean Accuracy": mean_accuracy,
        "Mean IoU": mean_IoU,
        "mean Precision": mean_precision,
        "mean Recall": mean_recall,
        "Mean F1 Score": mean_f1_score
    }

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(class_mapping)

    # Save metrics if out_name is provided
    if out_name:
        with open(out_name, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for metric_name, metric_value in metrics.items():
                writer.writerow([metric_name, metric_value])
        
        class_metrics_out_name = out_name.rsplit(".", 1)[0] + "_classwise." + out_name.rsplit(".", 1)[1]
        with open(class_metrics_out_name, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Class", "Accuracy", "IoU", "Precision", "Recall", "F1 Score"])
            for i in range(1, evaluator.num_class):
                class_name = class_mapping[i]
                writer.writerow([class_name, classwise_overal_accuracy[i], IoU[i], 
                                 precision[i], recall[i], f1_score[i]])

    return metrics
