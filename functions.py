import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import math

def box_plots(cols_number):
    cols_number.boxplot(figsize=(11, 6))
    #plt.yscale('log') # Makes the boxes more comparable while preserving relative differences.
    plt.title('Boxplots for All Numeric Columns')
    plt.xticks(rotation=90)
    plt.show()

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
                vmax=1
                mask=mask)
    
    plt.title('Correlation Heatmap of Numeric Columns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
