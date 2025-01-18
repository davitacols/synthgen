# Set the backend to 'Agg' before importing matplotlib
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from synthgen import SynthGen

def showcase_basic_generation():
    """Demonstrate basic dataset generation."""
    print("\n1. Basic Dataset Generation")
    print("-" * 50)
    
    gen = SynthGen(seed=42)
    df = gen.generate_tabular(rows=100, cols=3)
    print("\nBasic dataset with default parameters:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())

def showcase_mixed_columns():
    """Demonstrate generation of mixed column types."""
    print("\n2. Mixed Column Types")
    print("-" * 50)
    
    gen = SynthGen(seed=42)
    
    df = gen.generate_tabular(
        rows=200,
        cols=4,
        col_types=['numeric', 'categorical', 'numeric', 'categorical'],
        col_names=['price', 'category', 'quantity', 'status'],
        noise=0.1
    )
    
    print("\nMixed column types dataset:")
    print(df.head())
    print("\nColumn info:")
    print(df.info())
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    
    # Numeric columns
    sns.histplot(df['price'], bins=30, ax=axes[0, 0])
    axes[0, 0].set_title('Price Distribution')
    
    sns.histplot(df['quantity'], bins=30, ax=axes[0, 1])
    axes[0, 1].set_title('Quantity Distribution')
    
    # Categorical columns
    sns.countplot(data=df, x='category', ax=axes[1, 0])
    axes[1, 0].set_title('Category Distribution')
    
    sns.countplot(data=df, x='status', ax=axes[1, 1])
    axes[1, 1].set_title('Status Distribution')
    
    plt.tight_layout()
    plt.savefig('mixed_columns_distribution.png')
    plt.close()

def showcase_custom_parameters():
    """Demonstrate custom parameter settings."""
    print("\n3. Custom Parameters")
    print("-" * 50)
    
    gen = SynthGen(seed=42)
    gen.set_categorical_values(['Low', 'Medium', 'High'])
    
    df = gen.generate_tabular(
        rows=150,
        cols=3,
        col_types=['numeric', 'categorical', 'numeric'],
        col_names=['revenue', 'risk_level', 'cost'],
        numeric_params=[
            {'mean': 1000, 'std': 200, 'noise': 0.05},
            {'mean': 750, 'std': 150, 'noise': 0.05}
        ],
        categorical_params=[
            {'categories': ['Low', 'Medium', 'High'], 
             'probabilities': [0.2, 0.5, 0.3]}
        ]
    )
    
    print("\nCustom parameters dataset:")
    print(df.head())
    print("\nSummary by risk level:")
    print(df.groupby('risk_level').agg({
        'revenue': ['mean', 'std'],
        'cost': ['mean', 'std']
    }))
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='revenue', y='cost', hue='risk_level')
    plt.title('Revenue vs Cost by Risk Level')
    plt.savefig('revenue_cost_scatter.png')
    plt.close()

def showcase_time_series():
    """Demonstrate generating time series data."""
    print("\n4. Time Series Data")
    print("-" * 50)
    
    gen = SynthGen(seed=42)
    n_days = 365
    
    df = gen.generate_tabular(
        rows=n_days,
        cols=3,
        col_types=['numeric', 'numeric', 'numeric'],
        col_names=['daily_sales', 'temperature', 'foot_traffic'],
        numeric_params=[
            {'mean': 1000, 'std': 200, 'noise': 0.1},
            {'mean': 20, 'std': 5, 'noise': 0.05},
            {'mean': 500, 'std': 100, 'noise': 0.1}
        ]
    )
    
    df.index = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    print("\nTime series dataset:")
    print(df.head())
    print("\nMonthly averages:")
    print(df.resample('M').mean())
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))
    
    df['daily_sales'].plot(ax=ax1)
    ax1.set_title('Daily Sales')
    
    df['temperature'].plot(ax=ax2)
    ax2.set_title('Temperature')
    
    df['foot_traffic'].plot(ax=ax3)
    ax3.set_title('Foot Traffic')
    
    plt.tight_layout()
    plt.savefig('time_series.png')
    plt.close()

if __name__ == "__main__":
    print("SynthGen Feature Showcase")
    print("=" * 50)
    
    showcase_basic_generation()
    showcase_mixed_columns()
    showcase_custom_parameters()
    showcase_time_series()
    
    print("\nDemonstration complete! Check the generated PNG files for visualizations.")