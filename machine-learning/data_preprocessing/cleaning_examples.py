# -----------------------------
# Example 1: Missing Data Handling
# -----------------------------
data = {
    'A': [1, 2, np.nan, 4, np.nan],
    'B': [5, np.nan, 7, 8, 15],
    'C': [9, 10, 11, np.nan, np.nan]
}
df = pd.DataFrame(data)

print(df)
print(df.dropna(how='all'))

# -----------------------------
# Example 2: Standardization (Z-Score)
# -----------------------------
data_to_scale = np.array([[10], [50], [100], [150]])
scaler = StandardScaler()
scaler.fit(data_to_scale)
scaled_data_std = scaler.transform(data_to_scale)
print(scaled_data_std)

# -----------------------------
# Example 3: One-Hot Encoding
# -----------------------------
data_cat = {
    'Size': ['Small', 'Medium', 'Large', 'Small'],
    'Color': ['Red', 'Blue', 'Red', 'Green']
}
df_cat = pd.DataFrame(data_cat)
df_one_hot = pd.get_dummies(df_cat, columns=['Size'], prefix='Color')
print(df_one_hot)

# -----------------------------
# Example 4: Combined Cleaning & Scaling
# -----------------------------
data = {
    'Product_ID': [1,2,3,4,5,6,7,8],
    'Price': [10.5,25.0,np.nan,40.0,5.0,150.0,30.0,20.0],
    'Region': ['North','South','East','North',np.nan,'West','East','South'],
    'Sales': [100,200,50,400,75,500,np.nan,150]
}
df = pd.DataFrame(data)

df['Price'] = df['Price'].fillna(df['Price'].median())
df['Region'] = df['Region'].fillna(df['Region'].mode()[0])
df['Sales'] = df['Sales'].fillna(df['Sales'].mean())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(np.array(df[['Price']]))

df_cleaned = pd.get_dummies(df, columns=['Region'], prefix='Region')
print(df_cleaned.head())
df_cleaned.info()
