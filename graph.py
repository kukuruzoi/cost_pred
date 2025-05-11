import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('car_price_dataset.csv')

plt.figure(figsize=(10, 5))

# 1. Распределение целевой переменной (Price)
sns.histplot(df['Price'], bins=30)
plt.title('Распределение цен (Price)')
plt.savefig('graph/price_dist.png')

# 2. Распределение числовых признаков
numeric_cols = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count']
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, len(numeric_cols), i)
    sns.histplot(data=df, x=col, bins=20)
    plt.title(f'Распределение {col}')
plt.tight_layout()
plt.savefig('graph/numeric_dist.png')
plt.close()

# 3. Распределение категориальных признаков
categorical_cols = ['Model', 'Fuel_Type', 'Transmission']
plt.figure(figsize=(12, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    plot = sns.countplot(data=df, x=col)
    plot.set_xticklabels(plot.get_xticklabels(), 
                       rotation=90, 
                       ha='right', fontsize=6)
    plt.title(f'Распределение {col}')
plt.tight_layout()
plt.savefig('graph/categorical_dist.png')
plt.close()
