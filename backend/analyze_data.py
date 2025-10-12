import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('insurance.csv')

print('=== OUTLIER DETECTION (IQR Method) ===')
for col in ['age', 'bmi', 'children', 'charges']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f'{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)')
    print(f'  Range: [{df[col].min():.2f}, {df[col].max():.2f}]')
    print(f'  IQR bounds: [{lower:.2f}, {upper:.2f}]')

print('\n=== VALUE DISTRIBUTIONS ===')
print('Sex:', df['sex'].value_counts().to_dict())
print('Smoker:', df['smoker'].value_counts().to_dict())
print('Region:', df['region'].value_counts().to_dict())
print('Children:', df['children'].value_counts().to_dict())

print('\n=== Z-SCORE OUTLIERS (|z| > 3) ===')
for col in ['age', 'bmi', 'charges']:
    z_scores = np.abs(stats.zscore(df[col]))
    z_outliers = len(df[z_scores > 3])
    print(f'{col}: {z_outliers} extreme outliers')

print('\n=== DATA QUALITY CHECKS ===')
print('BMI outliers (extreme values):')
print(f'  BMI < 15: {len(df[df["bmi"] < 15])}')
print(f'  BMI > 50: {len(df[df["bmi"] > 50])}')

print('\nAge distribution:')
print(f'  Age < 18: {len(df[df["age"] < 18])}')
print(f'  Age > 64: {len(df[df["age"] > 64])}')

print('\nCharges anomalies:')
print(f'  Charges < 1000: {len(df[df["charges"] < 1000])}')
print(f'  Charges > 50000: {len(df[df["charges"] > 50000])}')

print('\n=== CORRELATION WITH TARGET ===')
print(df[['age', 'bmi', 'children', 'charges']].corr()['charges'].sort_values(ascending=False))
