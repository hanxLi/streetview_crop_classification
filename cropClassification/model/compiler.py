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
from eval import do_accuracy_evaluation
from trainval import train, validate
from predict_uncertain import *
from chipping import generate_stacked_image
from loss import *

class ModelCompiler:
    """
    ModelCompiler for managing segmentation with optional uncertainty-aware U-Net.
    """
    def __init__(self, model, params_init=None, freeze_params=None, 
                 resume_training=False, optimizer=None, scheduler=None):
        self.working_dir = os.getcwd()
        self.out_dir = "outputs"
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.gpu = torch.cuda.is_available()
        self.mps = torch.backends.mps.is_available()

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
            self.load_params(params_init, freeze_params, resume_training, optimizer, scheduler)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {num_params / 1e6:.1f}M")

    def load_params(self, dir_params, freeze_params=None, resume_training=False, 
                    optimizer=None, scheduler=None):
        print(f"Loading model parameters from: {dir_params}")
        checkpoint = torch.load(dir_params, map_location=torch.device('cpu'))

        model_state_dict = checkpoint.get('state_dict', checkpoint)
        if any(key.startswith("module.") for key in model_state_dict.keys()):
            model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

        self.model.load_state_dict(model_state_dict, strict=False)
        self.model = self.model.to(self.device)

        if freeze_params:
            for i, param in enumerate(self.model.parameters()):
                if i in freeze_params:
                    param.requires_grad = False

        print("Model parameters loaded successfully.")

        if resume_training and optimizer:
            if 'optimizer' in checkpoint:
                print("Loading optimizer state...")
                optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler and 'scheduler' in checkpoint:
                print("Loading scheduler state...")
                scheduler.load_state_dict(checkpoint['scheduler'])

    def fit(self, trainDataset, valDataset, epochs, optimizer_name, lr_init, 
            lr_policy, criterion, momentum=None, resume=False, resume_epoch=None, 
            log=True, use_ancillary=False, class_weights=None, **kwargs):
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
            resume: Whether to resume training from a checkpoint.
            resume_epoch: Epoch to resume from, if resuming.
            log: Whether to log metrics to TensorBoard.
            use_ancillary: Whether to use ancillary data.
            class_weights: Optional class weights (Tensor) to use with Focal Loss.
            **kwargs: Additional keyword arguments for the optimizer and scheduler.
        """

        if criterion == FocalLoss:
            print("Using Focal Loss with class weights." if class_weights is not None else "Using Focal Loss.")
            criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction='mean', ignore_index=-100)
        elif criterion == BalancedCrossEntropyLoss:
            print("Using Balanced Cross Entropy Loss with class weights." if class_weights is not None else "Using Balanced Cross Entropy Loss.")
            criterion = BalancedCrossEntropyLoss(class_weights=class_weights, ignore_index=-100, reduction='mean')
        elif criterion == AleatoricLoss:
            print("Using Aleatoric Loss.")
            criterion = AleatoricLoss(reduction='mean', ignore_index=-100)
        elif criterion == BalancedCrossEntropyUncertaintyLoss:
            print("Using Balanced Cross Entropy Uncertainty Loss.")
            criterion = BalancedCrossEntropyUncertaintyLoss(ignore_index=-100, reduction='mean')
        else:
            print("Using the user-provided criterion class.")
            criterion = criterion()  # Initialize any other criterion class without class weights

        # Set up model directories for checkpoints and logs
        self.model_dir = f"{self.working_dir}/{self.out_dir}/{self.model_name}_ep{epochs}"
        self.checkpoint_dir = Path(self.model_dir) / "chkpt"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        train_loss, val_loss = [], []

        print("-------------------------- Start training --------------------------")
        start = datetime.now()

        writer = SummaryWriter(log_dir=self.model_dir) if log else None

        # Initialize optimizer and scheduler
        optimizer = self.get_optimizer(optimizer_name, lr_init, momentum)
        scheduler = self.get_scheduler(optimizer, lr_policy, **kwargs)

        # Resume training from a checkpoint if specified
        if resume:
            resume_epoch = self.resume_checkpoint(optimizer, scheduler, resume_epoch)

        # Training loop
        for epoch in range(resume_epoch or 0, epochs):
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
                self.save_checkpoint(optimizer, scheduler, epoch + 1)

            print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds")

        # Save the final model checkpoint at the end of training
        self.save_checkpoint(optimizer, scheduler, epochs, final=True)

        if writer:
            writer.close()

        print(f"Training finished in {(datetime.now() - start).seconds}s")


    def save_checkpoint(self, optimizer, scheduler, epoch, final=False):
        """
        Save the current model checkpoint. If 'final' is True, save with a special name.
        """
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if final:
            save_path = self.checkpoint_dir / "final_checkpoint.pth.tar"
            print(f"Saving final model checkpoint: {save_path}")
        else:
            save_path = self.checkpoint_dir / f"{epoch}_checkpoint.pth.tar"

        torch.save(checkpoint, save_path)

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
        model_output_dir = Path(self.out_dir) / self.model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        # Associate the out_name with the model's output directory, if provided
        if out_name:
            out_name = model_output_dir / out_name
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

    def predict_and_display(self, image_path, csv_path, step=16, window_size=(224, 224), 
                                num_classes=3, save=False):
            """
            Predict on a full image, display the predicted mask overlaid on the original image, 
            and optionally save the results.

            Args:
                image_path (str): Path to the input image.
                csv_path (str): Path to the ancillary data CSV.
                step (int): Sliding window step size.
                window_size (tuple): Size of each sliding window patch.
                num_classes (int): Number of classes for segmentation.
                save (bool): Whether to save the prediction and overlay images.
            """
            print(f"Predicting on image: {image_path}")

            # Perform the prediction
            pred_mask, uncertainty_map = predict_full_image(
                self.model, image_path, csv_path, num_classes, self.device, step, window_size
            )

            # Overlay prediction on the original image and display
            overlay_image = overlay_prediction(image_path, pred_mask)
            Image.fromarray(overlay_image).show()

            # Optionally save the results
            if save:
                save_prediction_results(image_path, pred_mask, uncertainty_map, overlay_image)

            return pred_mask, uncertainty_map
    
    def simple_predict_and_display(self, image_path, csv_path, mean=None, std=None, step=16, window_size=(224, 224), 
                                num_classes=3, save=False):
        pred_mask = simple_predict_full_image(self.model, image_path, csv_path, num_classes, self.device, step, window_size)
        # pred_mask = optimized_predict_full_image(self.model, image_path, csv_path, num_classes, self.device, step, window_size)

        # Overlay prediction on the original image and display
        overlay_image = overlay_prediction(image_path, pred_mask)
        Image.fromarray(overlay_image).show()

        # Optionally save the results
        if save:
            save_prediction_results(image_path, pred_mask, None, overlay_image)

        return pred_mask
