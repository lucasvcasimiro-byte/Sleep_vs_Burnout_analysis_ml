import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import math
#histograms

usage= pd.read_csv("Smartphone_Usage_Productivity_Dataset_50000.csv", index_col=0)
usage.head()
cols_number = usage.select_dtypes(include=[np.number])

def box_plots(cols_number):
    cols_number.boxplot(figsize=(11, 6))
    #plt.yscale('log') # Makes the boxes more comparable while preserving relative differences.
    plt.title('Boxplots for All Numeric Columns')
    plt.xticks(rotation=90)
    plt.show()

def cor_heatmap(cols_number):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cols_number.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numeric Columns')
    plt.show()
