import os
import torch
import time
import torch.optim as optim
import cv2
from PIL import Image
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from eval import do_accuracy_evaluation
from trainval import train, validate
from predict_uncertain import predict_full_image, load_ancillary_data

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
            log=True, use_ancillary=False, **kwargs):
        """
        Train the model using the provided datasets, criterion, and optimizer.
        """
        self.model_dir = f"{self.working_dir}/{self.out_dir}/{self.model_name}_ep{epochs}"
        self.checkpoint_dir = Path(self.model_dir) / "chkpt"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        train_loss, val_loss = [], []

        print("-------------------------- Start training --------------------------")
        start = datetime.now()

        writer = SummaryWriter(log_dir=self.model_dir) if log else None

        optimizer = self.get_optimizer(optimizer_name, lr_init, momentum)
        scheduler = self.get_scheduler(optimizer, lr_policy, **kwargs)

        if resume:
            resume_epoch = self.resume_checkpoint(optimizer, scheduler, resume_epoch)

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

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(optimizer, scheduler, epoch + 1)

            print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds")

        if writer:
            writer.close()

        print(f"Training finished in {(datetime.now() - start).seconds}s")

    def save_checkpoint(self, optimizer, scheduler, epoch):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f"{epoch}_checkpoint.pth.tar")

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

    def evaluate(self, dataloader, num_classes, class_mapping, out_name=None, log_uncertainty=False):
        """
        Evaluate the model using the provided dataloader and compute metrics.
        """
        print("-------------------------- Start Evaluation --------------------------")
        metrics = do_accuracy_evaluation(
            self.model, dataloader, num_classes, class_mapping, 
            out_name=out_name, log_uncertainty=log_uncertainty
        )
        print("-------------------------- Evaluation Complete --------------------------")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        return metrics

def predict_image(self, image_path, csv_path, step=16, window_size=(224, 224)):
    """
    Predict on a full image and optionally generate an uncertainty map.
    """
    print(f"Predicting on image: {image_path}")

    # Load ancillary data associated with the image
    image_name = os.path.basename(image_path)
    ancillary_data = load_ancillary_data(csv_path, image_name)

    # Preprocess the input image
    input_image = self.preprocess_image(image_path)

    # Sliding window inference
    predictions, uncertainties = [], []
    for (x, y, patch) in self.sliding_window(input_image, step, window_size):
        pred, uncertainty = self.predict_patch(patch, ancillary_data)
        predictions.append((x, y, pred, uncertainty))
        uncertainties.append(uncertainty)

    # Stitch together predictions and uncertainties
    pred_mask, uncertainty_map = self.stitch_predictions(input_image.shape, window_size, step, predictions, uncertainties)

    # Draw bounding boxes on the original image
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    result_image = self.draw_bounding_boxes(original_image_rgb, pred_mask, uncertainty_map)

    # Display the final result
    Image.fromarray(result_image).show()

    print("Prediction completed.")
    return pred_mask, uncertainty_map

