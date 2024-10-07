import os
import torch
import torch.optim as optim
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from eval import do_accuracy_evaluation
from trainval import train, validate
from predict import *


class ModelCompiler:
    """
    Simplified ModelCompiler for managing segmentation model training, evaluation, and predictions.
    
    Args:
        model (torch.nn.Module): PyTorch model for segmentation.
        params_init (str, optional): Path to initial model parameters (checkpoint).
        freeze_params (list, optional): List of indices for parameters to keep frozen.
    """
    
    def __init__(self, model, params_init=None, freeze_params=None):
        self.working_dir = os.getcwd()
        self.out_dir = "outputs"
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.gpu = torch.cuda.is_available()
        self.mps = torch.backends.mps.is_available()

        # Determine which device to use: CUDA, MPS, or CPU
        if self.gpu:
            print('---------- GPU (CUDA) available ----------')
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
        elif self.mps:
            print('---------- MPS available ----------')
            print('Warning: MPS is still under optimization and might have performance issues.')
            self.device = torch.device('mps')
            self.model = self.model.to(self.device)
        else:
            print('---------- Using CPU ----------')
            print('Using CUDA or MPS is recommended for better performance.')
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)

        # Load initial model parameters if provided
        if params_init:
            self.load_params(params_init, freeze_params)

        # Count and display trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {num_params / 1e6:.1f}M")
        
        if params_init:
            print(f"---------- Pre-trained {self.model_name} model compiled successfully ----------")
        else:
            print(f"---------- {self.model_name} model compiled successfully ----------")
    
    def load_params(self, dir_params, freeze_params=None):
        """
        Load model parameters from a checkpoint file and freeze specified layers if needed.
        
        Args:
            dir_params (str): Path to the checkpoint file containing the model parameters.
            freeze_params (list, optional): List of parameter indices to freeze.
        """
        params_init = torch.load(dir_params)
        model_dict = self.model.state_dict()

        # Strip 'module.' from keys if the checkpoint was trained with DataParallel
        if "module" in list(params_init.keys())[0]:
            params_init = {k[7:]: v.cpu() for k, v in params_init.items() if k[7:] in model_dict}
        else:
            params_init = {k: v.cpu() for k, v in params_init.items() if k in model_dict}

        model_dict.update(params_init)
        self.model.load_state_dict(model_dict)

        # Optionally freeze layers
        if freeze_params:
            for i, p in enumerate(self.model.parameters()):
                if i in freeze_params:
                    p.requires_grad = False

    def fit(self, trainDataset, valDataset, epochs, optimizer_name, lr_init, lr_policy, 
            criterion, momentum=None, resume=False, resume_epoch=None, log=True, **kwargs):
        """
        Train the model with the given datasets, criterion, optimizer, and learning rate scheduler.
        
        Args:
            trainDataset (Dataset): Training dataset.
            valDataset (Dataset): Validation dataset.
            epochs (int): Number of training epochs.
            optimizer_name (str): Name of the optimizer to use ('SGD' or 'Adam').
            lr_init (float): Initial learning rate.
            lr_policy (str): Learning rate scheduler policy ('steplr' or 'multisteplr').
            criterion (torch.nn.Module): Loss function.
            momentum (float, optional): Momentum (for SGD optimizer).
            resume (bool, optional): Whether to resume training from a checkpoint.
            resume_epoch (int, optional): Epoch to resume from.
            log (bool, optional): Enable TensorBoard logging.
            **kwargs: Additional arguments for learning rate scheduler.
        """
        self.model_dir = f"{self.working_dir}/{self.out_dir}/{self.model_name}_ep{epochs}"
        self.checkpoint_dir = Path(self.model_dir) / "chkpt"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        train_loss = []
        val_loss = []
        print("-------------------------- Start training --------------------------")
        start = datetime.now()

        # Setup TensorBoard logging if enabled
        if log:
            writer = SummaryWriter(log_dir=self.model_dir)
        else:
            writer = None

        # Setup optimizer
        optimizer = self.get_optimizer(optimizer_name, lr_init, momentum)

        # Setup learning rate scheduler (StepLR or MultiStepLR)
        scheduler = self.get_scheduler(optimizer, lr_policy, **kwargs)

        # Optionally resume from checkpoint
        if resume:
            resume_epoch = self.resume_checkpoint(optimizer, scheduler, resume_epoch)

        # Main training loop
        for epoch in range(resume_epoch or 0, epochs):
            print(f"[{epoch+1}/{epochs}]")
            train(trainDataset, self.model, criterion, optimizer, scheduler,
                   trainLoss=train_loss, device=self.device)
            validate(valDataset, self.model, criterion, valLoss=val_loss,
                      device=self.device)

            # Step the scheduler
            if scheduler:
                scheduler.step()

            # Log learning rate if logging is enabled
            if log and scheduler:
                writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.save_checkpoint(optimizer, scheduler, epoch + 1)

        if writer:
            writer.close()
        print(f"-------------------------- Training finished in {(datetime.now() - start).seconds}s --------------------------")
        return train_loss, val_loss

    def get_optimizer(self, optimizer_name, lr_init, momentum=None):
        """
        Retrieve the optimizer based on the optimizer_name.
        
        Args:
            optimizer_name (str): Name of the optimizer ('SGD' or 'Adam').
            lr_init (float): Initial learning rate.
            momentum (float, optional): Momentum value (if using SGD).
        
        Returns:
            torch.optim.Optimizer: The optimizer for training.
        """
        if optimizer_name.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr_init, momentum=momentum, weight_decay=1e-4)
        elif optimizer_name.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=lr_init, weight_decay=1e-4)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def get_scheduler(self, optimizer, lr_policy, **kwargs):
        """
        Retrieve the learning rate scheduler based on the lr_policy (StepLR or MultiStepLR).
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to schedule learning rate.
            lr_policy (str): Learning rate scheduler policy ('steplr' or 'multisteplr').
            **kwargs: Additional arguments for the scheduler.
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: The learning rate scheduler.
        """
        lr_policy = lr_policy.lower()
        if lr_policy == "steplr":
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.5))
        elif lr_policy == "multisteplr":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=kwargs.get('milestones', [10, 20]), gamma=kwargs.get('gamma', 0.5))
        else:
            raise ValueError(f"Unsupported learning rate policy: {lr_policy}")
        
    def accuracy_evaluation(self, eval_dataset, filename):
        """
        Evaluate the accuracy of the model on the provided evaluation dataset.

        Args:
            eval_dataset (DataLoader): The evaluation dataset to evaluate the model on.
            filename (str): The filename to save the evaluation results in the output CSV.
    """

        if not os.path.exists(Path(self.working_dir) / self.out_dir):
            os.makedirs(Path(self.working_dir) / self.out_dir)

        os.chdir(Path(self.working_dir) / self.out_dir)

        print("---------------- Start evaluation ----------------")

        start = datetime.now()

        do_accuracy_evaluation(self.model, eval_dataset, num_classes, class_mapping, filename)

        duration_in_sec = (datetime.now() - start).seconds
        print(
            f"---------------- Evaluation finished in {duration_in_sec}s ----------------")

    def resume_checkpoint(self, optimizer, scheduler, resume_epoch):
        """
        Resume training from a checkpoint.
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer to load state.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to load state.
            resume_epoch (int, optional): The epoch to resume from.
        
        Returns:
            int: The epoch to resume training from.
        """
        checkpoint_path = self.checkpoint_dir / f"{resume_epoch}_checkpoint.pth.tar"
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            return checkpoint["epoch"]
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    def predict_regular_image(self, image_path, label_path=None, chip_size=224, overlap=32, class_num=3, output_path=None, plot=False):
        """
        Predict segmentation masks on regular images using a trained model.

        Args:
            image_path (str): Path to the input image.
            label_path (str, optional): Path to the ground truth label (optional, for plotting).
            chip_size (int): Size of the image chips for tiling.
            overlap (int): Overlap between neighboring chips.
            class_num (int): Number of classes for segmentation.
            output_path (str, optional): Path to save the predicted mask.
            plot (bool): Whether to plot the original image, label, and prediction.
        
        Returns:
            np.ndarray: The predicted segmentation mask.
        """
        # Device handling
        device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == 'cpu':
            print("Using CPU. It is recommended to use CUDA or MPS for better performance.")
        elif device.type == 'mps':
            print("Using MPS. Note: MPS may have performance issues.")

        # Perform prediction
        pred_mask = predict_image(image_path, self.model, chip_size=chip_size, overlap=overlap, class_num=class_num, device=device)

        # Save prediction if output_path is provided
        if output_path:
            save_prediction(pred_mask, output_path)
            print(f"Prediction saved at {output_path}")

        # Plot results if requested
        if plot and label_path:
            plot_prediction(image_path, label_path, pred_mask)

        return pred_mask

    def save_checkpoint(self, optimizer, scheduler, epoch):
        """
        Save a checkpoint of the current model, optimizer, and scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer to save state.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save state.
            epoch (int): The current epoch number.
        """
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_dir / f"{epoch}_checkpoint.pth.tar")
