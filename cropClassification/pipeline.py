import os
import pandas as pd
import random
from torch.utils.data import DataLoader
from pathlib import Path

from cropClassification.load_data import RoadsideCropImageDataset
from cropClassification.model_train.compiler import ModelCompiler
from cropClassification.model.unets import *


class PipelineManager:
    """
    Manager class for crop segmentation pipeline that orchestrates:
    - Model and dataset loading
    - Training, evaluation, and inference workflows
    - Integration with ModelCompiler for all core functionality
    """
    
    def __init__(self, config):
        """
        Initialize the pipeline manager with configuration.
        
        Args:
            config (dict): Configuration dictionary with model, dataset, and training parameters
        """
        self.config = config
        self.model_comp = None
        self.train_loader = None
        self.val_loader = None
        self.working_dir = config["model"]["working_dir"]
        print(f"Pipeline manager initialized with {config['model']['name']} model configuration")
    
    def _index_generator(self, df, count=3):
        """
        Generate random indices within the range of a DataFrame's length.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame for which to generate random indices
        count : int, optional
            The number of random indices to generate (default: 1)
        
        Returns:
        --------
        list
            A list of random integers between 0 and len(df)-1 (inclusive)
        """
        # Ensure the count doesn't exceed the DataFrame length
        max_count = min(count, len(df))
        
        # Generate the random indices
        if max_count == len(df):
            # If requesting all indices, return a shuffled list of all indices
            indices = list(range(len(df)))
            random.shuffle(indices)
            return indices[:count]
        else:
            # Otherwise, generate random indices
            return [random.randint(0, len(df) - 1) for _ in range(max_count)]

    def _load_dataset(self, classwise=False):
        """
        Internal method to load and prepare datasets from configuration.
        
        Args:
            classwise (bool): Whether to use class-wise normalization
        """
        train_df = pd.read_csv(self.config['dataset']['train_csv'])
        val_df = pd.read_csv(self.config['dataset']['val_csv'])

        # Set up common parameters
        train_params = {
            'dataframe': train_df,
            'root_dir': self.config['dataset']['train_root_path'],
            'usage': 'train',
            'use_ancillary': True,
            'ancillary_classes': 3
        }
        
        val_params = {
            'dataframe': val_df,
            'root_dir': self.config['dataset']['val_root_path'],
            'usage': 'val',
            'use_ancillary': True,
            'ancillary_classes': 3
        }
        
        # Add normalization parameters based on classwise flag
        if classwise:
            # Use class-wise normalization
            train_params['classwise_norm'] = self.config['dataset']['classwise_norm']
            val_params['classwise_norm'] = self.config['dataset']['classwise_norm']
        else:
            # Use global normalization
            train_params['mean'] = self.config['dataset']['train_mean']
            train_params['std'] = self.config['dataset']['train_std']
            val_params['mean'] = self.config['dataset']['val_mean']
            val_params['std'] = self.config['dataset']['val_std']
        
        # Create datasets
        train_dataset = RoadsideCropImageDataset(**train_params)
        val_dataset = RoadsideCropImageDataset(**val_params)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['validation']['batch_size'], 
            shuffle=False
        )
        
        print(f"Loaded datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    def _initialize_model(self, params_init=None, load_datasets=True, classwise_norm=False):
        """
        Initialize the model and optionally load datasets.
        
        Args:
            params_init (str, optional): Path to pre-trained model weights
            load_datasets (bool): Whether to load datasets
            classwise_norm (bool): Whether to use class-wise normalization
            
        Returns:
            self: For method chaining
        """
        model_name = self.config['model']['name']
        
        # Dynamically create the model using the specified name
        model = eval(model_name)(
            n_channels=self.config['model']['params']['in_channels'],
            n_classes=self.config['model']['params']['out_classes'],
            ancillary_data_dim=3,
            dropout_rate=self.config['training']['dropout_rate']
        )
        tmp_name = f"{model_name}_ep{self.config['training']['epochs']}_lr{self.config['training']['learning_rate']}_batch{self.config['training']['batch_size']}"
        
        # Create the model_comp with all required attributes
        self.model_comp = ModelCompiler(
            model=model, 
            working_dir=self.working_dir, 
            params_init=params_init, 
            save_name=tmp_name
        )
        
        # Explicitly set the predict_save_path attribute for inference
        self.model_comp.predict_save_path = Path(self.working_dir) / tmp_name / "inference"
        os.makedirs(self.model_comp.predict_save_path, exist_ok=True)
        
        # Also set model_dir explicitly
        self.model_comp.model_dir = f"{self.working_dir}/{tmp_name}/"
        
        if params_init:
            print(f"Loaded {model_name} model with pre-trained weights from {params_init}")
        else:
            print(f"Initialized {model_name} model with random weights")
        
        # Load datasets if requested
        if load_datasets:
            self._load_dataset(classwise=classwise_norm)
        
        return self
    
    def _train(self):
        """
        Train the model using configuration settings.
        
        Returns:
            self: For method chaining
        """
        if self.model_comp is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        if self.train_loader is None or self.val_loader is None:
            self._load_dataset(classwise=False)
            
        print(f"Starting model training for {self.config['training']['epochs']} epochs...")
        self.model_comp.fit(
            trainDataset=self.train_loader,
            valDataset=self.val_loader,
            epochs=self.config['training']['epochs'],
            optimizer_name=self.config['training']['optimizer']['type'],
            lr_init=self.config['training']['learning_rate'],
            lr_policy='steplr',
            criterion=self.config['training']['criterion'],
            class_weights=self.config['training']['classwise_weights'],
            log=True,
            use_ancillary=True,
            **self.config['training']['scheduler']['params']
        )
        
        print("Model training completed")
        return self
    
    def _evaluate(self, print_results=True):
        """
        Evaluate the model on validation data.
        
        Args:
            print_results (bool): Whether to print evaluation results summary
            
        Returns:
            tuple: Metrics from evaluation (aggregated_metrics, classwise_metrics)
        """
        if self.model_comp is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        if self.val_loader is None:
            self._load_dataset(classwise=False)
            
        print("Starting model evaluation...")
        metrics = self.model_comp.evaluate(
            dataloader=self.val_loader,
            num_classes=self.config['model']['params']['out_classes'],
            class_mapping=self.config['evaluation']['class_mapping'],
            out_name=self.config['evaluation']['filename'],
            log_uncertainty=False,
            return_val=True
        )
        
        # The print_results parameter controls whether to print a summary of evaluation metrics
        # The original metrics are still printed by the ModelCompiler's evaluate method
        if print_results and metrics:
            print("\nEvaluation Summary:")
            print(f"Overall Accuracy: {metrics[0]['Overall Accuracy']:.4f}")
            print(f"Mean IoU: {metrics[0]['Mean IoU']:.4f}")
            print(f"Mean F1 Score: {metrics[0]['Mean F1 Score']:.4f}")
        
        return metrics
    
    def _predict(self, summary_df, image_indices=None):
        """
        Run inference on images from the summary DataFrame.
        
        Args:
            summary_df (pd.DataFrame): DataFrame containing image metadata
            image_indices (list, optional): List of indices to process from summary_df. 
                                           If None, process all images.
            save_dir (str, optional): Directory to save inference results
            
        Returns:
            dict: Dictionary mapping image names to their prediction masks
        """
        if self.model_comp is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
            
        # Create the save directory if specified

        
        # Determine which images to process
        if image_indices is None:
            image_indices = list(range(len(summary_df)))
        
        # Store results
        results = {}
        
        # Process each image
        for i, idx in enumerate(image_indices):
            row = summary_df.iloc[idx]
            image_name = row['image_name']
            image_path = os.path.join(self.config["model"]["root_dir"], row['image_path'])
            
            print(f"Processing image {i+1}/{len(image_indices)}: {image_name}")
            
            try:
                # Get prediction mask using the ModelCompiler's simple_predict method
                pred_mask = self.model_comp.simple_predict(
                    image_path=image_path,
                    summary_df=summary_df,
                    norm_params=[self.config['dataset']['val_mean'], self.config['dataset']['val_std']]
                )
                
                results[image_name] = pred_mask
                
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")
                continue
        
        print(f"Completed inference on {len(results)} images")
        return results

    # Higher-level workflows for common scenarios
    
    def evaluate_pretrain(self, model_path):
        """
        Load a pre-trained model and evaluate it.
        
        Args:
            model_path (str): Path to the pre-trained model weights
            
        Returns:
            tuple: Evaluation metrics
        """
        print(f"Loading model from {model_path} and running evaluation...")
        self._initialize_model(params_init=model_path)
        return self._evaluate()
    
    def predict_pretrain(self, model_path, summary_df, image_indices=None):
        """
        Load a pre-trained model and run inference.
        
        Args:
            model_path (str): Path to the pre-trained model weights
            summary_df (pd.DataFrame): DataFrame with image metadata
            image_indices (list, optional): Specific images to process
            save_dir (str, optional): Directory to save results
            
        Returns:
            dict: Inference results
        """
        if image_indices is None:
            image_indices = self._index_generator(summary_df)
        print(f"Loading model from {model_path} and running inference...")
        self._initialize_model(params_init=model_path, load_datasets=False)
        return self._predict(summary_df, image_indices)
    
    def train(self):
        """
        Train a model from scratch and evaluate it.
        
        Returns:
            tuple: Evaluation metrics
        """
        print("Starting training and evaluation pipeline...")
        self._initialize_model()
        self._train()
        return self._evaluate()
    
    def train_and_predict(self, summary_df, image_indices=None):
        """
        Train a model from scratch and run inference.
        
        Args:
            summary_df (pd.DataFrame): DataFrame with image metadata
            image_indices (list, optional): Specific images to process
            save_dir (str, optional): Directory to save results
            
        Returns:
            dict: Inference results
        """
        print("Starting training and inference pipeline...")
        self._initialize_model()
        self._train()
        if image_indices is None:
            image_indices = self._index_generator(summary_df)
        return self._predict(summary_df, image_indices)
    
    def full_pipeline(self, summary_df, image_indices=None):
        """
        Run the complete pipeline: train, evaluate, and predict.
        
        Args:
            summary_df (pd.DataFrame): DataFrame with image metadata
            image_indices (list, optional): Specific images to process
            save_dir (str, optional): Directory to save results
            
        Returns:
            tuple: (evaluation_metrics, inference_results)
        """
        print("Starting full pipeline: training, evaluation, and inference...")
        self._initialize_model()
        self._train()
        metrics = self._evaluate()
        if image_indices is None:
            image_indices = self._index_generator(summary_df)
        results = self._predict(summary_df, image_indices)
        return metrics, results
    
    