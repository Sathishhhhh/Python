import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import HealthcarePreprocessor
from train_model import ModelTrainer
import pickle

class HealthcareEDA:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up matplotlib and seaborn styling"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def load_data(self):
        """Load data"""
        self.df = pd.read_csv(self.filepath)
        return self.df
    
    def basic_statistics(self):
        """Display basic statistics"""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
    
    def plot_distributions(self):
        """Plot distributions of numerical features"""
        print("\nGenerating distribution plots...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(numerical_cols):
            row = idx // 3
            col_idx = idx % 3
            ax = axes[row, col_idx]
            
            ax.hist(self.df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution of {col.capitalize()}')
            ax.set_xlabel(col.capitalize())
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/01_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/01_distributions.png")
        plt.close()
    
    def plot_categorical(self):
        """Plot categorical features"""
        print("Generating categorical plots...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 4))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        fig.suptitle('Categorical Features Distribution', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(categorical_cols):
            ax = axes[idx]
            value_counts = self.df[col].value_counts()
            ax.bar(value_counts.index, value_counts.values, color='lightcoral', edgecolor='black', alpha=0.7)
            ax.set_title(f'{col.capitalize()} Distribution')
            ax.set_xlabel(col.capitalize())
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/02_categorical.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/02_categorical.png")
        plt.close()
    
    def plot_correlations(self):
        """Plot correlation heatmap"""
        print("Generating correlation heatmap...")
        
        # Create numeric version for correlation
        df_numeric = self.df.copy()
        for col in df_numeric.select_dtypes(include=['object']).columns:
            df_numeric[col] = pd.Categorical(df_numeric[col]).codes
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df_numeric.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/03_correlation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/03_correlation.png")
        plt.close()
    
    def plot_features_vs_target(self):
        """Plot features vs target variable"""
        print("Generating feature vs target plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Features vs Premium Charges', fontsize=16, fontweight='bold')
        
        numerical_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                         if col != 'charges']
        
        for idx, col in enumerate(numerical_cols):
            row = idx // 3
            col_idx = idx % 3
            ax = axes[row, col_idx]
            
            ax.scatter(self.df[col], self.df['charges'], alpha=0.5, s=30)
            ax.set_title(f'{col.capitalize()} vs Charges')
            ax.set_xlabel(col.capitalize())
            ax.set_ylabel('Premium Charges ($)')
        
        plt.tight_layout()
        plt.savefig('plots/04_features_vs_target.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/04_features_vs_target.png")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all plots"""
        import os
        os.makedirs('plots', exist_ok=True)
        
        self.load_data()
        self.basic_statistics()
        self.plot_distributions()
        self.plot_categorical()
        self.plot_correlations()
        self.plot_features_vs_target()
        
        print("\n" + "="*60)
        print("✓ All plots generated successfully!")
        print("="*60)

if __name__ == "__main__":
    eda = HealthcareEDA('data/healthcare_data.csv')
    eda.generate_all_plots()
