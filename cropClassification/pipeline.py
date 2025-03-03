import os
import pandas as pd
import random
from torch.utils.data import DataLoader
from pathlib import Path

import optuna
import time
from datetime import datetime

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

    
    def hypertune(self, n_trials=20):
        """
        Perform hyperparameter tuning using Optuna.
        
        The search space is expected to be defined in the config under 'hypertuning.search_space'.
        Results are automatically saved to the working directory.
        
        Args:
            n_trials (int): Number of trials to run (default: 20)
            
        Returns:
            dict: Best parameters and their performance metrics
        """
        
        # Check if search space is defined in config
        if 'hypertuning' not in self.config or 'search_space' not in self.config['hypertuning']:
            raise ValueError("Search space must be defined in config['hypertuning']['search_space']")
        
        search_space = self.config['hypertuning']['search_space']
        
        # Create timestamp and directory for results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"hypertune_{timestamp}"
        results_dir = os.path.join(self.working_dir, "hypertuning", study_name)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Starting hyperparameter tuning with {n_trials} trials")
        print(f"Results will be saved to {results_dir}")
        print(f"Search space: {search_space}")
        
        # Define the objective function for Optuna
        def objective(trial):
            # Create hyperparameters for this trial
            params = {}
            
            for param_name, param_values in search_space.items():
                if isinstance(param_values, list) and len(param_values) > 0:
                    # Handle discrete values
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    # Handle continuous ranges
                    min_val, max_val = param_values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                    else:
                        # Float parameter (use log scale for learning rates)
                        log_scale = param_name == 'learning_rate'
                        params[param_name] = trial.suggest_float(param_name, min_val, max_val, log=log_scale)
                else:
                    raise ValueError(f"Invalid search space format for parameter '{param_name}'")
            
            # Update configuration with current hyperparameters
            if 'learning_rate' in params:
                self.config['training']['learning_rate'] = params['learning_rate']
            if 'drop_rate' in params:
                self.config['training']['dropout_rate'] = params['drop_rate']
            if 'epochs' in params:
                self.config['training']['epochs'] = params['epochs']
            if 'train_batch_size' in params:
                self.config['training']['batch_size'] = params['train_batch_size']
            
            # Log the current trial parameters
            print(f"\nTrial {trial.number}/{n_trials}:")
            for name, value in params.items():
                print(f"  {name}: {value}")
            
            start_time = time.time()
            
            try:
                # Initialize the model with the current hyperparameters
                self._initialize_model(load_datasets=True)
                
                # Train the model
                self._train()
                
                # Evaluate the model
                agg_metrics, class_metrics = self._evaluate(print_results=True)
                
                # Calculate training time
                elapsed_time = time.time() - start_time
                
                # Extract the metric we want to optimize (mean IoU)
                metric = agg_metrics['Mean IoU']
                
                # Save trial results
                trial_result = {
                    'trial': trial.number,
                    **params,
                    'mean_iou': metric,
                    'overall_accuracy': agg_metrics['Overall Accuracy'],
                    'mean_f1': agg_metrics['Mean F1 Score'],
                    'background_iou': class_metrics['Background']['IoU'],
                    'maize_iou': class_metrics['Maize']['IoU'],
                    'soybean_iou': class_metrics['Soybean']['IoU'],
                    'training_time': elapsed_time,
                    'model_dir': self.model_comp.model_dir
                }
                
                # Save trial to CSV
                trial_df = pd.DataFrame([trial_result])
                trial_csv = os.path.join(results_dir, f"trial_{trial.number}.csv")
                trial_df.to_csv(trial_csv, index=False)
                
                # Also append to all trials CSV
                all_trials_csv = os.path.join(results_dir, "all_trials.csv")
                if os.path.exists(all_trials_csv):
                    all_trials_df = pd.read_csv(all_trials_csv)
                    all_trials_df = pd.concat([all_trials_df, trial_df], ignore_index=True)
                else:
                    all_trials_df = trial_df
                all_trials_df.to_csv(all_trials_csv, index=False)
                
                print(f"Trial {trial.number} completed with Mean IoU: {metric:.4f} (time: {elapsed_time:.2f}s)")
                
                return metric  # Return the metric to be maximized
                
            except Exception as e:
                print(f"Error during trial {trial.number}: {e}")
                # Save the error information
                error_info = {
                    'trial': trial.number,
                    **params,
                    'error': str(e),
                    'training_time': time.time() - start_time
                }
                pd.DataFrame([error_info]).to_csv(
                    os.path.join(results_dir, f"trial_{trial.number}_error.csv"), 
                    index=False
                )
                # Return a very low score for failed trials
                return 0.0
        
        # Create the Optuna study
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # We want to maximize the IoU
            storage=f"sqlite:///{os.path.join(results_dir, 'optuna.db')}",
            load_if_exists=True
        )
        
        # Run the optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Get the best parameters and value
        best_params = study.best_params
        best_value = study.best_value
        
        print("\nHyperparameter tuning completed!")
        print("\nBest Parameters:")
        for name, value in best_params.items():
            print(f"  {name}: {value}")
        print(f"Best Mean IoU: {best_value:.4f}")
        
        # Save the best parameters
        with open(os.path.join(results_dir, "best_parameters.txt"), 'w') as f:
            f.write(f"Best Mean IoU: {best_value:.4f}\n\n")
            for param_name, param_value in best_params.items():
                f.write(f"{param_name}: {param_value}\n")
        
        # Generate and save visualizations
        # Export study visualizations
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(results_dir, "optimization_history.png"))
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(results_dir, "param_importances.png"))
        
        fig3 = optuna.visualization.plot_slice(study)
        fig3.write_image(os.path.join(results_dir, "slice_plot.png"))
        
        # Update the config with the best parameters
        if 'learning_rate' in best_params:
            self.config['training']['learning_rate'] = best_params['learning_rate']
        if 'drop_rate' in best_params:
            self.config['training']['dropout_rate'] = best_params['drop_rate']
        if 'epochs' in best_params:
            self.config['training']['epochs'] = best_params['epochs']
        if 'train_batch_size' in best_params:
            self.config['training']['batch_size'] = best_params['train_batch_size']
        
        print("\nConfig updated with best parameters!")
        
        # Return best parameters and value
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
    }