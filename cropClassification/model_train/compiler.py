import os
import torch
import time
import torch.optim as optim
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from cropClassification.model_eval.eval import do_accuracy_evaluation
from cropClassification.model_eval.val import validate
from cropClassification.model_inference.predict import simple_predict_full_image, overlay_prediction, save_prediction_results
from cropClassification.model_train.train import train
from cropClassification.utils.chipping import generate_stacked_image
from cropClassification.model.losses import *

class ModelCompiler:
    """
    ModelCompiler for managing segmentation with optional uncertainty-aware U-Net.
    """
    def __init__(self, model, working_dir, params_init=None, freeze_params=None, save_name=None):
        self.working_dir = working_dir
        self.out_dir = str(save_name)
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.gpu = torch.cuda.is_available()
        self.mps = torch.backends.mps.is_available()
        self.predict_save_path = None

        # Device configuration
        if self.gpu:
            print('---------- GPU (CUDA) available ----------')
            self.device = torch.device('cuda')
        elif self.mps:
            print('---------- MPS available ----------')
            print('Warning: MPS may have performance issues.')
            self.device = torch.device('mps')
        else:
            print('---------- Using CPU ----------')
            self.device = torch.device('cpu')

        self.model = self.model.to(self.device)

        if params_init:
            self.load_params(params_init, freeze_params)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {num_params / 1e6:.1f}M")

    def load_params(self, model_weights_path, freeze_params=None):
        """
        Load the model's weights and optionally freeze layers.

        Args:
            model_weights_path (str): Path to the saved model weights.
            freeze_params (list, optional): List of layer names to freeze.
        """
        print(f"Loading model weights from: {model_weights_path}")

        # Ensure self.device is a torch.device object
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device

        # Load the model weights
        model_state_dict = torch.load(model_weights_path, map_location=self.device, weights_only=True)

        # Detect and remove "module." prefix if loading from a Distributed Data Parallel (DDP) model
        if list(model_state_dict.keys())[0].startswith("module."):
            print("🛠 Detected 'module.' prefix in state dict, removing it...")
            model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}

        # Load the state dict and log missing/unexpected keys
        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)

        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")

        # Move model to the correct device
        self.model = self.model.to(self.device)
        print("Model weights loaded successfully.")

        # Optional: Freeze specific layers if requested
        if freeze_params:
            for name, param in self.model.named_parameters():
                if name in freeze_params:
                    param.requires_grad = False
                    print(f"Froze layer: {name}")

    def fit(self, trainDataset, valDataset, epochs, optimizer_name, lr_init, 
            lr_policy, criterion, momentum=None, log=True, use_ancillary=False, class_weights=None, **kwargs):
        """
        Train the model using the provided datasets, criterion, and optimizer.

        Args:
            trainDataset: DataLoader for training.
            valDataset: DataLoader for validation.
            epochs: Total number of epochs.
            optimizer_name: Name of the optimizer to use (e.g., 'adam', 'sgd').
            lr_init: Initial learning rate.
            lr_policy: Learning rate scheduler policy (e.g., 'steplr').
            criterion: Loss function provided by the user (e.g., CrossEntropyLoss).
            momentum: Momentum (for optimizers that require it).
            log: Whether to log metrics to TensorBoard.
            use_ancillary: Whether to use ancillary data.
            class_weights: Optional class weights (Tensor) to use with Focal Loss.
            **kwargs: Additional keyword arguments for the optimizer and scheduler.
        """

        if criterion == "FocalLoss":
            print("Using Focal Loss with class weights." if class_weights is not None else "Using Focal Loss.")
            criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction='mean', ignore_index=-100)
        elif criterion == "BalancedCrossEntropyLoss":
            print("Using Balanced Cross Entropy Loss with class weights." if class_weights is not None else "Using Balanced Cross Entropy Loss.")
            criterion = BalancedCrossEntropyLoss(class_weights=class_weights, ignore_index=-100, reduction='mean')
        elif criterion == "AleatoricLoss":
            print("Using Aleatoric Loss.")
            criterion = AleatoricLoss(reduction='mean', ignore_index=-100)
        elif criterion == "BalancedCrossEntropyUncertaintyLoss":
            print("Using Balanced Cross Entropy Uncertainty Loss.")
            criterion = BalancedCrossEntropyUncertaintyLoss(ignore_index=-100, reduction='mean')
        else:
            print("Using the user-provided criterion class.")
            criterion = criterion()  # Initialize any other criterion class without class weights

        # Set up model directories for checkpoints and logs
        self.model_dir = f"{self.working_dir}/{self.out_dir}/"
        self.checkpoint_dir = Path(self.model_dir) / "ckpts"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        train_loss, val_loss = [], []

        print("-------------------------- Start training --------------------------")
        start = datetime.now()

        writer = SummaryWriter(log_dir=self.model_dir) if log else None

        # Initialize optimizer and scheduler
        optimizer = self.get_optimizer(optimizer_name, lr_init, momentum)
        scheduler = self.get_scheduler(optimizer, lr_policy, **kwargs)

        # Resume training from a checkpoint if specified


        # Training loop
        for epoch in range(0, epochs):
            print(f"----------------------- [{epoch+1}/{epochs}] -----------------------")
            epoch_start = time.time()

            # Training step
            train_loss_epoch = train(
                trainDataset, self.model, criterion, optimizer, scheduler,
                trainLoss=train_loss, device=self.device, use_ancillary=use_ancillary
            )
            train_loss.append(train_loss_epoch)

            # Validation step
            val_loss_epoch = validate(
                valDataset, self.model, criterion, valLoss=val_loss,
                device=self.device, use_ancillary=use_ancillary
            )
            val_loss.append(val_loss_epoch)

            if scheduler:
                scheduler.step()

            # Log to TensorBoard
            if writer:
                writer.add_scalar('Loss/Train', train_loss_epoch, epoch)
                writer.add_scalar('Loss/Validation', val_loss_epoch, epoch)
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

            print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds")

        # Save the final model checkpoint at the end of training
        self.save_checkpoint(epochs, final=True)

        if writer:
            writer.close()

        print(f"Training finished in {(datetime.now() - start).seconds}s")
    
    def save_checkpoint(self, epoch, final=False):
        """
        Save the model's weights.
        """
        if final:
            save_path = self.checkpoint_dir / "final_model_weights.pth"
            print(f"Saving final model weights to: {save_path}")
        else:
            save_path = self.checkpoint_dir / f"ep{epoch}_model_weights.pth"

        torch.save(self.model.state_dict(), save_path)
        # print(f"Model weights saved to: {save_path}")

    def get_optimizer(self, optimizer_name, lr_init, momentum=None):
        if optimizer_name.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr_init, momentum=momentum, weight_decay=1e-4)
        elif optimizer_name.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr_init, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def get_scheduler(self, optimizer, lr_policy, **kwargs):
        if lr_policy.lower() == "steplr":
            return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))
        elif lr_policy.lower() == "multisteplr":
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs.get('milestones', [10, 20]), gamma=kwargs.get('gamma', 0.1))
        else:
            raise ValueError(f"Unsupported learning rate policy: {lr_policy}")
        
    def evaluate(self, dataloader, num_classes, class_mapping, out_name=None, 
                log_uncertainty=False, return_val=False):
        """
        Evaluate the model using the provided dataloader and compute metrics.
        """
        print("-------------------------- Start Evaluation --------------------------")

        # Create a folder for the model's results inside 'out_dir'
        model_eval_dir = Path(self.model_dir) / "eval"
        model_eval_dir.mkdir(parents=True, exist_ok=True)

        # Associate the out_name with the model's output directory, if provided
        if out_name:
            out_name = model_eval_dir / out_name
            print(f"Saving metrics to: {out_name}")

        # Perform the evaluation and collect metrics
        metrics, classwise_metrics = do_accuracy_evaluation(
            self.model, dataloader, num_classes, class_mapping, 
            out_name=out_name, log_uncertainty=log_uncertainty
        )

        print("-------------------------- Evaluation Complete --------------------------")

        # Print aggregated metrics
        print("Aggregated Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        # Print class-wise metrics
        print("\nClass-Wise Metrics:")
        for class_name, class_metrics in classwise_metrics.items():
            metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in class_metrics.items()])
            print(f"{class_name}: {metrics_str}")

        if return_val:
            return metrics, classwise_metrics
    
    def simple_predict(self, image_path, summary_df, norm_params, step=16, window_size=(224, 224), 
                    num_classes=3):
        """
        Predict segmentation mask for a single image using sliding window.
        
        Args:
            image_path (str): Path to the input image
            summary_df (pd.DataFrame): DataFrame containing image metadata
            norm_params (tuple): Mean and standard deviation for normalization
            step (int): Step size for sliding window
            window_size (tuple): Size of the sliding window
            num_classes (int): Number of segmentation classes
            save_path (str or bool): Path to save results, or False to not save
        
        Returns:
            np.ndarray: Predicted segmentation mask
        """

        self.predict_save_path = Path(self.model_dir) / "inference"

        os.makedirs(self.predict_save_path, exist_ok=True)

        pred_mask = simple_predict_full_image(
            self.model, 
            image_path, 
            summary_df, 
            num_classes, 
            self.device, 
            norm_params, 
            step, 
            window_size
        )

        # Overlay prediction on the original image and display
        overlay_image = overlay_prediction(image_path, pred_mask)
        Image.fromarray(overlay_image).show()

        save_prediction_results(
            image_path, 
            pred_mask, 
            overlay_image=overlay_image,
            save_dir=str(self.predict_save_path)
        )

        return pred_mask

