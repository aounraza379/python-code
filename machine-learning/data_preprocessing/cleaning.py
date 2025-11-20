import numpy as np
import pandas as pd

# Sample dataset
data = {
    'ID': [1, 2, 2, 3, 4, 5, np.nan],
    'Name': ['aoun', 'bb', 'bb', 'cc', 'ee', 'aa', 'bb'],
    'Age': [22, 25, 25, 23, np.nan, 30, 25],
    'Email': [
        'aa@gmail.com', 'bb@gmail.com', 'bb@gmail.com',
        'cc@gmail', 'ee@gmail.com', 'aa@gmail.com', np.nan
    ],
    'City': ['Lhr', 'Fsd', 'Fsd', 'Lhr', 'Isb', 'lahore', 'Faisalabad'],
    'Salary': [50000, 60000, 60000, 800000, 55000, 50000, 60000]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Remove duplicates
df = df.drop_duplicates()
print("\nAfter dropping duplicates:\n", df)

# Fill missing Age with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())
print("\nAfter filling missing Age:\n", df)

# Drop rows where ID or Email is missing
df.dropna(subset=['ID', 'Email'], inplace=True)
print("\nAfter dropping rows with missing ID or Email:\n", df)

# Standardize Email
df['Email'] = df['Email'].apply(lambda x: x if '.com' in x else x + '.com')
print("\nAfter standardizing Email:\n", df)

# Standardize City names
df['City'] = df['City'].str.lower()
df['City'] = df['City'].replace({
    'faisalabd': 'fsd',
    'fsd': 'fsd',
    'islamabad': 'isb',
    'isb': 'isb',
    'lahore': 'lhr',
    'lhr': 'lhr',
})

df['City'] = df['City'].str.title()
print("\nAfter cleaning City names:\n", df)

# Gender standardization
dt = {
    'Name': ['A', 'S', 'J', 'M', 'U'],
    'Gender': ['M', 'f', 'M ALE', 'Female', 'male']
}

d = pd.DataFrame(dt)

# Convert to lowercase
d['Gender'] = d['Gender'].str.lower()

# Replace variations with consistent values
d['Gender'] = d['Gender'].replace({
    'm ale': 'male',
    'm': 'male',
    'f': 'female',
    'mal': 'male',
    'fem': 'female'
})

# Swap case for demonstration
d['Gender'] = d['Gender'].str.swapcase()
print("\nStandardized Gender DataFrame:\n", d)

# Handle salary outliers
median_salary = df['Salary'].median()
print("\nMedian Salary:", median_salary)

df.loc[df['Salary'] > 200000, 'Salary'] = median_salary
print("\nAfter adjusting salary outliers:\n", df)
