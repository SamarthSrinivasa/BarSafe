#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
from tqdm.auto import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import from your existing safety filter script
from Bar_safe import SafetyScorer

# Hugging Face datasets
from datasets import load_dataset, disable_progress_bars
disable_progress_bars()

class JailbreakDatasetAnalyzer:
    def __init__(
        self, 
        dataset_name: str = "JailbreakV-28K/JailBreakV-28k", 
        output_dir: str = None,
        image_base_dir: str = None,
        max_samples: int = 75  # Limit to exactly 75 samples
    ):
        """
        Initialize the Jailbreak Dataset Analyzer
        
        Args:
            dataset_name (str): Hugging Face dataset name
            output_dir (str): Directory to save output files
            image_base_dir (str): Base directory where images are stored locally
            max_samples (int): Maximum number of samples to analyze
        """
        # Set output directory (default to script location)
        self.output_dir = output_dir or os.path.join(project_root, 'benchmark_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set image base directory
        self.image_base_dir = image_base_dir or project_root
        logger.info(f"Using image base directory: {self.image_base_dir}")
        
        # Store max samples parameter
        self.max_samples = max_samples
        
        # Load the dataset with specific configuration
        try:
            self.dataset = load_dataset(
                dataset_name, 
                "JailBreakV_28K",  # Use the full dataset
                split="JailBreakV_28K"
            )
            logger.info(f"Successfully loaded full dataset with {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Initialize safety scorer from your existing script
        self.safety_scorer = SafetyScorer()
        
        # Print dataset information
        self._print_dataset_info()
        self._validate_dataset()
    
    def _print_dataset_info(self):
        """
        Print detailed information about the dataset
        """
        logger.info("\n===== DATASET INFORMATION =====")
        logger.info(f"Total samples: {len(self.dataset)}")
        logger.info(f"Column names: {self.dataset.column_names}")
        
        # Print sample details
        logger.info("\n===== SAMPLE DETAILS =====")
        sample = self.dataset[0]
        for key, value in sample.items():
            logger.info(f"{key}: {type(value)}")
            # Print first few characters if it's a string
            if isinstance(value, str):
                logger.info(f"  Preview: {value[:100]}...")
    
    def _validate_dataset(self):
        """
        Validate the dataset structure and find matching samples for existing images
        """
        logger.info("\n===== VALIDATING DATASET =====")
        
        # Check for required fields
        required_fields = ['jailbreak_query', 'image_path']
        missing_fields = [field for field in required_fields if field not in self.dataset.column_names]
        if missing_fields:
            logger.warning(f"Missing required fields in dataset: {missing_fields}")
            raise ValueError(f"Required fields missing: {missing_fields}")
        
        # First, scan the directories to find actual images
        image_dirs = ['llm_transfer_attack', 'query_related', 'figstep']
        found_images = {}
        
        logger.info(f"Scanning directories for existing images: {', '.join(image_dirs)}")
        for dir_name in image_dirs:
            dir_path = os.path.join(self.image_base_dir, dir_name)
            if not os.path.exists(dir_path):
                logger.warning(f"Directory does not exist: {dir_path}")
                continue
                
            # Get all image files in this directory
            img_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            logger.info(f"Found {len(img_files)} images in {dir_name}")
            
            for img_file in tqdm(img_files, desc=f"Scanning {dir_name}"):
                # Store with full relative path
                relative_path = os.path.join(dir_name, img_file)
                found_images[relative_path] = os.path.join(dir_path, img_file)
                
                # Also store by filename for partial matching
                found_images[img_file] = os.path.join(dir_path, img_file)
        
        logger.info(f"Found {len(found_images) // 2} unique images across all directories")
        
        # Now, match with dataset entries (but only until we find exactly max_samples)
        valid_samples_idx = []
        match_counts = {'exact': 0, 'filename': 0}
        
        logger.info(f"Matching images with dataset entries (until we find exactly {self.max_samples} valid samples)")
        
        # First try exact path matches
        for i, sample in enumerate(tqdm(self.dataset, desc="Looking for exact path matches")):
            if not sample.get('jailbreak_query') or not sample.get('image_path'):
                continue
                
            # Try exact path match
            img_path = sample['image_path']
            if img_path in found_images:
                valid_samples_idx.append(i)
                match_counts['exact'] += 1
            
            # Stop if we've found enough
            if len(valid_samples_idx) >= self.max_samples:
                logger.info(f"Found {self.max_samples} exact path matches, stopping search")
                break
        
        # If we still need more, try filename matches
        if len(valid_samples_idx) < self.max_samples:
            remaining_needed = self.max_samples - len(valid_samples_idx)
            logger.info(f"Need {remaining_needed} more matches, trying filename matching")
            
            # Keep track of samples we've already added
            added_indices = set(valid_samples_idx)
            
            for i, sample in enumerate(tqdm(self.dataset, desc="Looking for filename matches")):
                # Skip if we already added this sample
                if i in added_indices:
                    continue
                    
                if not sample.get('jailbreak_query') or not sample.get('image_path'):
                    continue
                    
                # Try filename match
                img_filename = os.path.basename(sample['image_path'])
                if img_filename in found_images:
                    valid_samples_idx.append(i)
                    match_counts['filename'] += 1
                    added_indices.add(i)
                
                # Stop if we've found enough
                if len(valid_samples_idx) >= self.max_samples:
                    logger.info(f"Found {self.max_samples} total matches, stopping search")
                    break
        
        # Ensure we have exactly max_samples
        if len(valid_samples_idx) > self.max_samples:
            logger.info(f"Found {len(valid_samples_idx)} samples, limiting to exactly {self.max_samples}")
            valid_samples_idx = valid_samples_idx[:self.max_samples]
        elif len(valid_samples_idx) < self.max_samples:
            logger.warning(f"Only found {len(valid_samples_idx)} valid samples, fewer than requested {self.max_samples}")
        
        logger.info(f"Found {len(valid_samples_idx)} matching samples:")
        logger.info(f"  - Exact path matches: {match_counts['exact']}")
        logger.info(f"  - Filename matches: {match_counts['filename']}")
        
        # Create a mapping from dataset paths to actual file paths
        self.path_mapping = {}
        for idx in valid_samples_idx:
            img_path = self.dataset[idx]['image_path']
            if img_path in found_images:
                self.path_mapping[img_path] = found_images[img_path]
            else:
                # Must be a filename match
                img_filename = os.path.basename(img_path)
                if img_filename in found_images:
                    self.path_mapping[img_path] = found_images[img_filename]
                else:
                    logger.warning(f"Could not find matching image for {img_path}")
        
        # Store valid sample indices for later use
        self.valid_samples_idx = valid_samples_idx
        
        # Log some example valid samples
        logger.info("Sample of valid entries:")
        for i, idx in enumerate(valid_samples_idx[:5]):  # Show first 5 valid samples
            sample = self.dataset[idx]
            orig_path = sample['image_path']
            if orig_path in self.path_mapping:
                matched_path = self.path_mapping[orig_path]
                logger.info(f"Sample {i+1}: Index {idx}")
                logger.info(f"  - Dataset path: {orig_path}")
                logger.info(f"  - Matched to: {matched_path}")
            else:
                logger.warning(f"Sample {i+1}: Index {idx} - No mapping found for {orig_path}")
        
        # Ask user to proceed
        if not valid_samples_idx:
            logger.error("No valid samples with images found. Cannot proceed.")
            sys.exit(1)
            
        proceed = input(f"Found {len(valid_samples_idx)} valid samples. Proceed with analysis? (y/n): ")
        if proceed.lower() != 'y':
            logger.info("User chose not to proceed. Exiting.")
            sys.exit(0)
            
        logger.info(f"Proceeding with analysis of {len(valid_samples_idx)} samples")
    
    def get_full_image_path(self, relative_path):
        """
        Get the actual image path using our mapping from dataset validation
        """
        if hasattr(self, 'path_mapping') and relative_path in self.path_mapping:
            return self.path_mapping[relative_path]
            
        # Fallback: try direct path
        direct_path = os.path.join(self.image_base_dir, relative_path)
        if os.path.exists(direct_path):
            return direct_path
            
        # Try filename matching as last resort
        basename = os.path.basename(relative_path)
        for dir_name in ['llm_transfer_attack', 'query_related', 'figstep']:
            check_path = os.path.join(self.image_base_dir, dir_name, basename)
            if os.path.exists(check_path):
                return check_path
                
        logger.warning(f"Could not find image: {relative_path}")
        return None
    
    def analyze_text_toxicity(self):
        """
        Analyze text toxicity across validated samples with images
        
        Returns:
            Dict of average toxicity scores and DataFrame of all scores
        """
        logger.info("\n===== TEXT TOXICITY ANALYSIS =====")
        
        # Initialize toxicity storage
        toxicity_scores = {}
        all_scores = []
        
        # Analyze only validated samples with progress bar
        for idx in tqdm(self.valid_samples_idx, desc="Analyzing text toxicity"):
            try:
                # Get the sample
                sample = self.dataset[idx]
                
                # Score text
                text_scores = self.safety_scorer.score_text(sample['jailbreak_query'])
                
                # Save all scores for this sample
                sample_scores = {
                    'sample_id': idx, 
                    'jailbreak_query': sample['jailbreak_query'][:100] + '...',  # Truncate for readability
                    'image_path': sample['image_path'],
                    **text_scores
                }
                all_scores.append(sample_scores)
                
                # Aggregate scores
                for category, score in text_scores.items():
                    if category not in toxicity_scores:
                        toxicity_scores[category] = []
                    toxicity_scores[category].append(score)
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                logger.error(f"Sample content preview: {str(sample.get('jailbreak_query', ''))[:50]}")
        
        # Calculate average toxicity
        avg_toxicity = {}
        for category, scores in toxicity_scores.items():
            if scores:  # Only calculate if we have scores
                avg_toxicity[category] = np.mean(scores)
                logger.info(f"{category} toxicity: {avg_toxicity[category]:.4f}")
        
        # Create DataFrame with all scores
        scores_df = pd.DataFrame(all_scores)
        
        # Create a histogram for each toxicity category
        self._plot_toxicity_distributions(toxicity_scores)
        
        return avg_toxicity, scores_df
    
    def _plot_toxicity_distributions(self, toxicity_scores):
        """
        Plot distributions of toxicity scores across categories
        """
        # Setup the plot
        categories = list(toxicity_scores.keys())
        num_cats = len(categories)
        
        if num_cats == 0:
            logger.warning("No toxicity categories to plot")
            return
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(1, num_cats, figsize=(5*num_cats, 6))
        if num_cats == 1:
            axes = [axes]  # Make it iterable for single category
        
        # Plot each category
        for i, category in enumerate(categories):
            scores = toxicity_scores[category]
            ax = axes[i]
            
            # Plot histogram
            sns.histplot(scores, bins=20, kde=True, ax=ax)
            ax.set_title(f"{category} Distribution")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            
            # Add mean line
            mean_val = np.mean(scores)
            ax.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.3f}')
            
            # Add median line
            median_val = np.median(scores)
            ax.axvline(median_val, color='green', linestyle='-', 
                       label=f'Median: {median_val:.3f}')
            
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'toxicity_distributions.png'))
        plt.close(fig)
        logger.info(f"Saved toxicity distribution plots to {self.output_dir}")
    
    def safety_constraint_benchmark(
        self, 
        C_values: List[float] = [0.5, 1.0, 2.0],
        lambda_values: List[float] = [0.1, 1.0, 2.0]
    ):
        """
        Comprehensive safety constraint benchmarking with pre-computed scores
        
        Args:
            C_values (List[float]): Threshold ceiling values to test
            lambda_values (List[float]): Adaptive sensitivity values to test
        
        Returns:
            List of configuration results and DataFrame with detailed sample results
        """
        logger.info("\n===== SAFETY CONSTRAINT BENCHMARK =====")
        
        # Pre-compute scores for all samples first to avoid redundant API calls
        logger.info("Pre-computing toxicity scores for all samples")
        sample_scores = {}
        
        for idx in tqdm(self.valid_samples_idx, desc="Computing scores"):
            sample = self.dataset[idx]
            
            try:
                # Score text (only once)
                text_scores = self.safety_scorer.score_text(sample['jailbreak_query'])
                max_text_score = max(text_scores.values()) if text_scores else 0
                
                # Process image (only once)
                image_scores = {}
                max_image_score = 0
                image_path = None
                image_processed = False
                
                try:
                    image_path = self.get_full_image_path(sample['image_path'])
                    if image_path:
                        image_scores = self.safety_scorer.score_image(image_path)
                        max_image_score = max(image_scores.values()) if image_scores else 0
                        image_processed = True
                    else:
                        logger.warning(f"Could not find image for sample {idx}: {sample['image_path']}")
                except Exception as e:
                    logger.error(f"Error processing image for sample {idx}: {str(e)}")
                
                # Store all scores for this sample
                sample_scores[idx] = {
                    'text_scores': text_scores,
                    'max_text_score': max_text_score,
                    'image_scores': image_scores,
                    'max_image_score': max_image_score,
                    'image_path': image_path,
                    'image_processed': image_processed
                }
                
            except Exception as e:
                logger.error(f"Error computing scores for sample {idx}: {str(e)}")
                # Mark this sample as problematic
                sample_scores[idx] = None
        
        # Track results for different configurations
        logger.info("Evaluating safety constraint across parameter combinations")
        config_results = []
        detailed_results = []
        
        # Process each configuration using the pre-computed scores
        for C in C_values:
            for lambda_val in lambda_values:
                logger.info(f"Testing configuration C={C}, λ={lambda_val}")
                
                # Reset per configuration
                config_result = {
                    'C': C,
                    'lambda': lambda_val,
                    'total_samples': len(self.valid_samples_idx),
                    'processed_samples': 0,
                    'filtered_samples': 0,
                    'filtering_rate': 0,
                }
                
                # Use pre-computed scores for evaluation
                filtered_toxicity = []
                config_sample_results = []
                
                for idx in tqdm(self.valid_samples_idx, desc=f"Evaluating C={C}, λ={lambda_val}"):
                    # Skip samples with no scores
                    if idx not in sample_scores or sample_scores[idx] is None:
                        continue
                    
                    scores = sample_scores[idx]
                    sample_result = {
                        'sample_id': idx,
                        'C': C,
                        'lambda': lambda_val,
                    }
                    
                    # Add text scores
                    sample_result.update({
                        'max_text_score': scores['max_text_score'],
                        **{f"text_{k}": v for k, v in scores['text_scores'].items()}
                    })
                    
                    # Add image scores if available
                    if scores['image_processed']:
                        sample_result.update({
                            'image_processed': True,
                            'image_path': scores['image_path'],
                            'max_image_score': scores['max_image_score'],
                            **{f"image_{k}": v for k, v in scores['image_scores'].items()}
                        })
                    else:
                        sample_result['image_processed'] = False
                    
                    # Apply safety constraint
                    is_safe = self.safety_scorer.safety_constraint(
                        scores['text_scores'], 
                        scores['image_scores'],
                        C=C, 
                        lambda_val=lambda_val
                    )
                    
                    sample_result['is_safe'] = is_safe
                    config_result['processed_samples'] += 1
                    
                    # Track filtered samples
                    if not is_safe:
                        config_result['filtered_samples'] += 1
                        filtered_toxicity.append(max(scores['max_text_score'], scores['max_image_score']))
                        sample_result['filtered'] = True
                    else:
                        sample_result['filtered'] = False
                    
                    # Add to detailed results
                    config_sample_results.append(sample_result)
                
                # Calculate filtering rate based on processed samples
                if config_result['processed_samples'] > 0:
                    config_result['filtering_rate'] = (
                        config_result['filtered_samples'] / config_result['processed_samples']
                    ) * 100
                
                # Average toxicity of filtered samples
                if filtered_toxicity:
                    config_result['filtered_samples_avg_toxicity'] = np.mean(filtered_toxicity)
                    config_result['filtered_samples_max_toxicity'] = np.max(filtered_toxicity)
                    config_result['filtered_samples_min_toxicity'] = np.min(filtered_toxicity)
                else:
                    config_result['filtered_samples_avg_toxicity'] = 0
                    config_result['filtered_samples_max_toxicity'] = 0
                    config_result['filtered_samples_min_toxicity'] = 0
                
                logger.info(f"Configuration C={C}, λ={lambda_val}:")
                logger.info(f"  Processed Samples: {config_result['processed_samples']}/{config_result['total_samples']}")
                logger.info(f"  Filtered Samples: {config_result['filtered_samples']}")
                logger.info(f"  Filtering Rate: {config_result['filtering_rate']:.2f}%")
                logger.info(f"  Avg Toxicity of Filtered Samples: {config_result['filtered_samples_avg_toxicity']:.4f}")
                
                config_results.append(config_result)
                detailed_results.extend(config_sample_results)
        
        # Convert detailed results to DataFrame
        detailed_df = pd.DataFrame(detailed_results)
        logger.info(f"Processed {len(sample_scores)} samples successfully across {len(C_values) * len(lambda_values)} configurations")
        
        return config_results, detailed_df
    
    def visualize_benchmark_results(self, results, detailed_df):
        """
        Create comprehensive visualizations of benchmark results
        
        Args:
            results: List of configuration results
            detailed_df: DataFrame with detailed sample results
        """
        logger.info("\n===== VISUALIZING BENCHMARK RESULTS =====")
        
        if not results or len(results) == 0:
            logger.warning("No results to visualize")
            return
            
        if detailed_df is None or detailed_df.empty:
            logger.warning("No detailed results to visualize")
            return
        
        # Prepare data for visualization
        df = pd.DataFrame(results)
        
        # 1. Filtering Rate Heatmap
        try:
            plt.figure(figsize=(10, 6))
            plt.title('Safety Filter Performance: Filtering Rate (%)')
            pivot_table = df.pivot_table(
                index='C', 
                columns='lambda', 
                values='filtering_rate'
            )
            sns.heatmap(
                pivot_table, 
                annot=True, 
                cmap='YlOrRd',
                fmt='.2f'
            )
            plt.xlabel('Lambda (Adaptive Sensitivity)')
            plt.ylabel('C (Threshold Ceiling)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'filtering_rate_heatmap.png'))
            plt.close()
            logger.info("Generated filtering rate heatmap")
        except Exception as e:
            logger.error(f"Error generating filtering rate heatmap: {e}")
        
        # 2. Filtered Samples Toxicity - Scatter Plot
        try:
            plt.figure(figsize=(10, 6))
            plt.title('Toxicity vs Filtering Rate by Configuration')
            
            # Check if we have the necessary columns
            if 'filtering_rate' in df.columns and 'filtered_samples_avg_toxicity' in df.columns:
                scatter = plt.scatter(
                    df['filtering_rate'], 
                    df['filtered_samples_avg_toxicity'], 
                    c=df['C'],  # Color by C value
                    s=df['lambda'] * 100,  # Size by lambda
                    cmap='viridis',
                    alpha=0.7
                )
                
                # Add labels for each point
                for i, row in df.iterrows():
                    plt.annotate(
                        f"C={row['C']}, λ={row['lambda']}",
                        (row['filtering_rate'], row['filtered_samples_avg_toxicity']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8
                    )
                
                plt.xlabel('Filtering Rate (%)')
                plt.ylabel('Average Toxicity of Filtered Samples')
                plt.colorbar(scatter, label='C (Threshold Ceiling)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'filtered_samples_toxicity.png'))
                plt.close()
                logger.info("Generated filtered samples toxicity plot")
            else:
                logger.warning("Missing required columns for toxicity scatter plot")
        except Exception as e:
            logger.error(f"Error generating toxicity scatter plot: {e}")
        
        # 3. Calculate and visualize Attack Success Rate (ASR)
        try:
            # Add ASR to the DataFrame
            df['attack_success_rate'] = 100 - df['filtering_rate']
            
            # ASR Heatmap
            plt.figure(figsize=(10, 6))
            plt.title('Attack Success Rate (ASR) by Parameter Configuration')
            pivot_table = df.pivot_table(
                index='C', 
                columns='lambda', 
                values='attack_success_rate'
            )
            sns.heatmap(
                pivot_table, 
                annot=True, 
                cmap='YlOrRd_r',  # Reversed colormap: dark red = low ASR = good
                fmt='.2f'
            )
            plt.xlabel('Lambda (Adaptive Sensitivity)')
            plt.ylabel('C (Threshold Ceiling)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'asr_heatmap.png'))
            plt.close()
            logger.info("Generated ASR heatmap")
            
            # ASR vs Filtering Rate comparison
            plt.figure(figsize=(12, 8))
            plt.title('Defensive Performance: Filtering Rate vs Attack Success Rate')
            
            # Sort by filtering rate
            df_sorted = df.sort_values('filtering_rate', ascending=False)
            
            # Create bar chart
            bar_width = 0.35
            indices = np.arange(len(df_sorted))
            
            # Create labels for x-axis
            config_labels = [f"C={row['C']}, λ={row['lambda']}" for _, row in df_sorted.iterrows()]
            
            plt.bar(indices, df_sorted['filtering_rate'], bar_width, label='Filtering Rate (%)', color='blue')
            plt.bar(indices + bar_width, df_sorted['attack_success_rate'], bar_width, label='Attack Success Rate (%)', color='red')
            
            plt.xlabel('Parameter Configuration')
            plt.ylabel('Percentage (%)')
            plt.title('Filtering Rate vs. Attack Success Rate by Configuration')
            plt.xticks(indices + bar_width / 2, config_labels, rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'filtering_vs_asr.png'))
            plt.close()
            logger.info("Generated filtering rate vs ASR comparison")
        except Exception as e:
            logger.error(f"Error generating ASR visualizations: {e}")
        
        # 4. Text/Image Toxicity Distribution for Filtered vs Unfiltered
        try:
            if 'filtered' in detailed_df.columns and 'max_text_score' in detailed_df.columns:
                # Combine all configurations for an overall view
                plt.figure(figsize=(12, 6))
                filtered = detailed_df[detailed_df['filtered'] == True]
                unfiltered = detailed_df[detailed_df['filtered'] == False]
                
                # Plot distributions if we have data
                if not filtered.empty:
                    sns.kdeplot(filtered['max_text_score'], label='Filtered', color='red', fill=True, alpha=0.3)
                if not unfiltered.empty:
                    sns.kdeplot(unfiltered['max_text_score'], label='Unfiltered (ASR)', color='blue', fill=True, alpha=0.3)
                
                plt.title('Overall Text Toxicity Distribution: Filtered vs Unfiltered')
                plt.xlabel('Text Toxicity Score')
                plt.ylabel('Density')
                plt.legend()
                
                # Add overall ASR
                overall_asr = 100 * len(unfiltered) / len(detailed_df) if len(detailed_df) > 0 else 0
                plt.annotate(f'Overall ASR: {overall_asr:.1f}%', xy=(0.05, 0.95), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'overall_text_toxicity_distribution.png'))
                plt.close()
                logger.info("Generated text toxicity distribution plot")
                
                # If we have image scores, plot those too
                if 'max_image_score' in detailed_df.columns and detailed_df['max_image_score'].notna().any():
                    plt.figure(figsize=(12, 6))
                    
                    # Filter to rows with valid image scores
                    filtered_img = filtered[filtered['max_image_score'].notna()]
                    unfiltered_img = unfiltered[unfiltered['max_image_score'].notna()]
                    
                    if not filtered_img.empty:
                        sns.kdeplot(filtered_img['max_image_score'], label='Filtered', color='red', fill=True, alpha=0.3)
                    if not unfiltered_img.empty:
                        sns.kdeplot(unfiltered_img['max_image_score'], label='Unfiltered (ASR)', color='blue', fill=True, alpha=0.3)
                    
                    plt.title('Overall Image Toxicity Distribution: Filtered vs Unfiltered')
                    plt.xlabel('Image Toxicity Score')
                    plt.ylabel('Density')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'overall_image_toxicity_distribution.png'))
                    plt.close()
                    logger.info("Generated image toxicity distribution plot")
            else:
                logger.warning("Missing required columns for toxicity distribution plots")
        except Exception as e:
            logger.error(f"Error generating toxicity distribution plots: {e}")
    
    def generate_benchmark_report(self, results, detailed_df):
        """
        Generate a comprehensive benchmark report
        
        Args:
            results: List of configuration results
            detailed_df: DataFrame with detailed sample results
        
        Returns:
            Dict containing benchmark summary and detailed results
        """
        logger.info("\n===== GENERATING BENCHMARK REPORT =====")
        
        if not results or len(results) == 0:
            logger.warning("No results to include in report")
            return {"benchmark_summary": {}, "detailed_results": []}
            
        # Create DataFrame for easy analysis
        df = pd.DataFrame(results)
        
        # Calculate Attack Success Rate (ASR)
        df['attack_success_rate'] = 100 - df['filtering_rate']
        
        # Initialize report
        report = {
            'benchmark_summary': {
                'total_configurations': len(results),
                'total_samples': len(self.valid_samples_idx)
            },
            'detailed_results': results
        }
        
        # Find best configurations if we have results
        if not df.empty:
            # Best filtering rate
            if 'filtering_rate' in df.columns:
                best_idx = df['filtering_rate'].idxmax()
                if not pd.isna(best_idx):
                    report['benchmark_summary']['best_filtering_rate'] = df.loc[best_idx].to_dict()
            
            # Lowest ASR (best defense)
            if 'attack_success_rate' in df.columns:
                lowest_asr_idx = df['attack_success_rate'].idxmin()
                if not pd.isna(lowest_asr_idx):
                    report['benchmark_summary']['lowest_asr_config'] = df.loc[lowest_asr_idx].to_dict()
            
            # Most toxic filtered
            if 'filtered_samples_avg_toxicity' in df.columns:
                df_with_filtering = df[df['filtered_samples_avg_toxicity'] > 0]
                if not df_with_filtering.empty:
                    most_toxic_idx = df_with_filtering['filtered_samples_avg_toxicity'].idxmax()
                    if not pd.isna(most_toxic_idx):
                        report['benchmark_summary']['most_toxic_filtered'] = df_with_filtering.loc[most_toxic_idx].to_dict()
        
        # Add additional metrics from detailed results
        if detailed_df is not None and not detailed_df.empty:
            try:
                # How many samples were flagged by at least one configuration
                unique_filtered = detailed_df[detailed_df['filtered'] == True]['sample_id'].nunique()
                total_unique_samples = detailed_df['sample_id'].nunique()
                
                report['benchmark_summary']['unique_samples_filtered'] = unique_filtered
                report['benchmark_summary']['total_unique_samples'] = total_unique_samples
                report['benchmark_summary']['overall_filtering_rate'] = (
                    100 * unique_filtered / total_unique_samples if total_unique_samples > 0 else 0
                )
                report['benchmark_summary']['overall_asr'] = (
                    100 - report['benchmark_summary']['overall_filtering_rate']
                )
            except Exception as e:
                logger.error(f"Error calculating additional metrics: {e}")
        
        # Save report to JSON
        try:
            report_path = os.path.join(self.output_dir, 'jailbreak_benchmark_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Benchmark report generated and saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving benchmark report: {e}")
        
        # Save detailed results to CSV if available
        if detailed_df is not None and not detailed_df.empty:
            try:
                detailed_path = os.path.join(self.output_dir, 'detailed_benchmark_results.csv')
                detailed_df.to_csv(detailed_path, index=False)
                logger.info(f"Detailed results saved to {detailed_path}")
            except Exception as e:
                logger.error(f"Error saving detailed results: {e}")
        
        return report


def main():
    """
    Main function to run the benchmark
    """
    try:
        # Set paths
        output_dir = os.path.join(project_root, 'benchmark_results')
        image_base_dir = project_root  # Change this to your images directory
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Print banner
        print("\n" + "="*80)
        print(f"JAILBREAK DATASET SAFETY FILTER BENCHMARK")
        print("="*80)
        print(f"- Output directory: {output_dir}")
        print(f"- Image base directory: {image_base_dir}")
        print(f"- Target: Analyze exactly 75 samples")
        print("="*80 + "\n")
        
        # Initialize the dataset analyzer
        analyzer = JailbreakDatasetAnalyzer(
            output_dir=output_dir,
            image_base_dir=image_base_dir,
            max_samples=75  # Limit to exactly 75 samples
        )
        
        # Analyze text toxicity for validated samples
        avg_toxicity, text_scores_df = analyzer.analyze_text_toxicity()
        
        # Run safety constraint benchmark with parameter combinations
        results, detailed_df = analyzer.safety_constraint_benchmark(
            C_values=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
            lambda_values=[0.1, 0.25, 0.75, 0.5, 1.0]
        )
        
        # Visualize the results
        analyzer.visualize_benchmark_results(results, detailed_df)
        
        # Generate comprehensive report
        report = analyzer.generate_benchmark_report(results, detailed_df)
        
        # Print key findings
        print("\n\n" + "="*80)
        print("BENCHMARK KEY FINDINGS")
        print("="*80)
        
        if 'best_filtering_rate' in report['benchmark_summary']:
            best = report['benchmark_summary']['best_filtering_rate']
            print(f"\nBest Configuration (Highest Filtering Rate):")
            print(f"  - C = {best.get('C')}, λ = {best.get('lambda')}")
            print(f"  - Filtering Rate: {best.get('filtering_rate', 0):.2f}%")
            print(f"  - Samples Filtered: {best.get('filtered_samples', 0)}/{best.get('processed_samples', 0)}")
        
        if 'lowest_asr_config' in report['benchmark_summary']:
            best_defense = report['benchmark_summary']['lowest_asr_config']
            print(f"\nBest Defensive Configuration (Lowest Attack Success Rate):")
            print(f"  - C = {best_defense.get('C')}, λ = {best_defense.get('lambda')}")
            print(f"  - Attack Success Rate: {best_defense.get('attack_success_rate', 0):.2f}%")
        
        if 'overall_asr' in report['benchmark_summary']:
            print(f"\nOverall Results:")
            print(f"  - Overall Filtering Rate: {report['benchmark_summary'].get('overall_filtering_rate', 0):.2f}%")
            print(f"  - Overall Attack Success Rate: {report['benchmark_summary'].get('overall_asr', 0):.2f}%")
            print(f"  - Unique Samples Filtered: {report['benchmark_summary'].get('unique_samples_filtered', 0)}/{report['benchmark_summary'].get('total_unique_samples', 0)}")
        
        print("\nAnalysis complete! Results saved to:", output_dir)
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()