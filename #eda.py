#eda

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
User_ID – Unique identifier

Age – User age (18–60)

Gender – Male, Female, Other

Occupation – Student, Professional, Freelancer, Business Owner

Device_Type – Android / iOS

Daily_Phone_Hours – Average daily phone usage

Social_Media_Hours – Daily time spent on social media

Work_Productivity_Score – Productivity score (1–10)

Sleep_Hours – Average sleep duration

Stress_Level – Stress rating (1–10)

App_Usage_Count – Number of apps used daily

Caffeine_Intake_Cups – Daily caffeine consumption

Weekend_Screen_Time_Hours – Screen time during weekends
"""

def categorical_cols(cat_cols, dataset):
    """
    Shows categorical columns values distributions (identifies all unique values)
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 
    for ax, col in zip(axes, categorical_cols): 
        sns.countplot(data=dataset, x=col, ax=ax, color='#0EE071') 
        ax.set_title(f'Distribution of {col}') 
    plt.tight_layout() 
    plt.show() 


