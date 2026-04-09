import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import math

def corr_heatmap(dataset, cols):
    """
    Display correlation heatmap for numerical columns
    """
    # Create correlation matrix
    corr_matrix = dataset[cols].corr()

    # Create mask for heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create figure and heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'},
                vmin=-1,
                vmax=1,
                mask=mask)
    
    plt.title('Correlation Heatmap of Numeric Columns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def outlier_detection(dataset, columns=None):
    """
    Display boxplot distributions for numerical columns.
    
    """
    
    # Select columns
    if columns is None:
        columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        if 'user_id' in columns:
            columns.remove('user_id')
    
    # Visualization
    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    
    # Flatten axes array for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        ax.boxplot(dataset[col].dropna())
        ax.set_title(f'{col}')
        ax.set_ylabel(col)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def check_duplicates(dataset, subset=None):
    """
    Check for duplicate rows in the dataset.
    """
    
    # Count duplicates
    num_duplicates = dataset.duplicated(subset=subset).sum()
    print(f"Duplicate rows: {num_duplicates}")

def categorical_distributions(dataset, columns=None):
    """
    Display distribution of categorical variables.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        The dataset to visualize
    columns : list, optional
        Specific columns to analyze. If None, uses all categorical columns
    """
    
    # Select columns
    if columns is None:
        columns = dataset.select_dtypes(include=[object]).columns.tolist()
    
    if len(columns) == 0:
        print("No categorical columns found")
        return
    
    # Visualization
    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    
    # Flatten axes array for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        dataset[col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'{col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_distribution_grid(df, columns):
    """Plots a grid of histograms to see data spread."""
    plt.figure(figsize=(10, 7))
    for i, col in enumerate(columns, 1):
        plt.subplot(4, 4, i)
        sns.histplot(df[col], kde=True)

        plt.title(f'Distribution of {col}', fontsize=10)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Generates a clean, masked heatmap."""
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.show()

def plot_scatter_insight(df, x_col, y_col, hue_col):
    """Helps visualize potential clusters before running algorithms."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.5)
    plt.title(f"{x_col} vs {y_col} by {hue_col}")
    plt.show()

