import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class CreditRiskEDA:
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
        print("\n" + "="*70)
        print("CREDIT RISK DATASET - BASIC STATISTICS")
        print("="*70)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nClass Distribution:")
        print(self.df['Default'].value_counts())
        print(f"Default Rate: {self.df['Default'].mean():.2%}")
    
    def plot_distributions(self):
        """Plot distributions of numerical features"""
        print("\nGenerating distribution plots...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'Default']
        
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            ax.hist(self.df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('plots/01_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/01_distributions.png")
        plt.close()
    
    def plot_categorical(self):
        """Plot categorical features"""
        print("Generating categorical plots...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(12, 4))
        if len(categorical_cols) == 1:
            axes = [axes]
        
        fig.suptitle('Categorical Features Distribution', fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(categorical_cols):
            ax = axes[idx]
            value_counts = self.df[col].value_counts()
            ax.bar(value_counts.index, value_counts.values, color='lightcoral', edgecolor='black', alpha=0.7)
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
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
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = df_numeric.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/03_correlation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/03_correlation.png")
        plt.close()
    
    def plot_default_analysis(self):
        """Plot default vs features"""
        print("Generating default analysis plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        fig.suptitle('Feature Analysis by Default Status', fontsize=16, fontweight='bold')
        
        features = ['Age', 'Income', 'Credit_Score', 'Loan_Amount', 'Num_Defaults_Past', 'Employment_Years']
        
        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            self.df.boxplot(column=feature, by='Default', ax=ax)
            ax.set_title(f'{feature} by Default Status')
            ax.set_xlabel('Default (0=No, 1=Yes)')
            ax.set_ylabel(feature)
        
        plt.suptitle('', fontsize=1)  # Remove auto title
        plt.tight_layout()
        plt.savefig('plots/04_default_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/04_default_analysis.png")
        plt.close()
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        print("Generating class distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Count plot
        default_counts = self.df['Default'].value_counts()
        colors = ['green', 'red']
        axes[0].bar(['Non-Default', 'Default'], default_counts.values, color=colors, alpha=0.7, edgecolor='black')
        axes[0].set_title('Default Count')
        axes[0].set_ylabel('Count')
        
        # Percentage plot
        default_pct = self.df['Default'].value_counts(normalize=True) * 100
        axes[1].pie(default_pct.values, labels=['Non-Default', 'Default'], autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('Default Percentage')
        
        plt.tight_layout()
        plt.savefig('plots/05_class_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: plots/05_class_distribution.png")
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
        self.plot_default_analysis()
        self.plot_class_distribution()
        
        print("\n" + "="*70)
        print("✓ All plots generated successfully!")
        print("="*70)

if __name__ == "__main__":
    eda = CreditRiskEDA('data/credit_risk_data.csv')
    eda.generate_all_plots()
