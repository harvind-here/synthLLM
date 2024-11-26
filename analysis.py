import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the JSON data
with open('medical_billing_dataset.json', 'r') as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(data)

# Convert numeric columns to float
df['total_charge'] = df['total_charge'].astype(float)
df['insurance_claim_amount'] = df['insurance_claim_amount'].astype(float)

# Plot Gaussian distribution for total charge
plt.figure(figsize=(10, 6))
sns.histplot(df['total_charge'], kde=True)
plt.title('Gaussian Distribution of Total Charges')
plt.xlabel('Total Charge ($)')
plt.ylabel('Frequency')
plt.savefig('total_charge_distribution.png')
plt.close()

# Plot Gaussian distribution for insurance claim amount
plt.figure(figsize=(10, 6))
sns.histplot(df['insurance_claim_amount'], kde=True)
plt.title('Gaussian Distribution of Insurance Claim Amounts')
plt.xlabel('Insurance Claim Amount ($)')
plt.ylabel('Frequency')
plt.savefig('insurance_claim_distribution.png')
plt.close()

# Plot Total Charge vs Insurance Claim Amount
plt.figure(figsize=(10, 6))
plt.scatter(df['total_charge'], df['insurance_claim_amount'])
plt.title('Total Charge vs Insurance Claim Amount')
plt.xlabel('Total Charge ($)')
plt.ylabel('Insurance Claim Amount ($)')
plt.savefig('charge_vs_claim.png')
plt.close()

# Calculate correlation
correlation = df['total_charge'].corr(df['insurance_claim_amount'])

print(f"Correlation between Total Charge and Insurance Claim Amount: {correlation:.2f}")

# Basic statistics
print("\nBasic Statistics:")
print(df[['total_charge', 'insurance_claim_amount']].describe())

# Check for Gaussian distribution
_, p_value_charge = stats.normaltest(df['total_charge'])
_, p_value_insurance = stats.normaltest(df['insurance_claim_amount'])

print(f"\np-value for Total Charge normality test: {p_value_charge:.4f}")
print(f"p-value for Insurance Claim Amount normality test: {p_value_insurance:.4f}")

