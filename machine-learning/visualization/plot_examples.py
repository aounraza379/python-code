# -----------------------------
# Example 1: High Score Analysis
# -----------------------------
data = {
    'Name': ['Kaye','Bryan','Chris','Delidamer','Eve','Farah'],
    'Age': [24,27,22,30,25,29],
    'Score': [95.5,88.0,76.2,99.9,82.5,91.0]
}
df_data = pd.DataFrame(data)
df_data["Normalized_Score"] = round(df_data["Score"]/100,2)
df_high_score = df_data[df_data["Score"]>=85]

plt.figure(figsize=(8,5))
plt.bar(x=df_high_score["Name"], height=df_high_score["Score"], color='teal')
plt.title('High Score Analysis (Score >= 85)')
plt.xlabel("Student Name")
plt.ylabel("Score")
plt.ylim(80,105)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
