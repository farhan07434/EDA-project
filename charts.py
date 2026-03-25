import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load merged dataset
df = pd.read_csv("merged_dataset.csv")

sns.set_style("whitegrid")

# ------------------------------
# UBM: UNDERSTAND (Data Overview)
# ------------------------------

# 1. Distribution of Release Years
plt.figure()
sns.histplot(df['release_year'], bins=30)
plt.title("Distribution of Release Years")
plt.show()

# 2. Runtime Distribution
plt.figure()
sns.histplot(df['runtime'], bins=30)
plt.title("Movie Runtime Distribution")
plt.show()

# 3. TMDB Score Distribution
plt.figure()
sns.histplot(df['tmdb_score'], bins=30)
plt.title("TMDB Score Distribution")
plt.show()

# ------------------------------
# UBM: BUILD (Relationships)
# ------------------------------

# 4. Popularity vs Score
plt.figure()
sns.scatterplot(x='tmdb_popularity', y='tmdb_score', data=df)
plt.title("Popularity vs Score")
plt.show()

# 5. Runtime vs Score
plt.figure()
sns.scatterplot(x='runtime', y='tmdb_score', data=df)
plt.title("Runtime vs Score")
plt.show()

# 6. Release Year vs Popularity
plt.figure()
sns.scatterplot(x='release_year', y='tmdb_popularity', data=df)
plt.title("Release Year vs Popularity")
plt.show()

# 7. Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# ------------------------------
# UBM: MEASURE (Insights)
# ------------------------------

# 8. Average Score by Year
plt.figure()
df.groupby('release_year')['tmdb_score'].mean().plot()
plt.title("Average Score Over Years")
plt.show()

# 9. Top Popular Movies
plt.figure()
df.nlargest(10, 'tmdb_popularity').plot(
    x='title', y='tmdb_popularity', kind='bar')
plt.title("Top 10 Popular Movies")
plt.xticks(rotation=45)
plt.show()

# 10. Runtime Boxplot
plt.figure()
sns.boxplot(x=df['runtime'])
plt.title("Runtime Outliers")
plt.show()

# 11. Score Boxplot
plt.figure()
sns.boxplot(x=df['tmdb_score'])
plt.title("Score Distribution Boxplot")
plt.show()

# 12. Popularity Density Plot
plt.figure()
sns.kdeplot(df['tmdb_popularity'], fill=True)
plt.title("Popularity Density")
plt.show()

# 13. Runtime Density Plot
plt.figure()
sns.kdeplot(df['runtime'], fill=True)
plt.title("Runtime Density")
plt.show()

# 14. Score vs Year Trend
plt.figure()
sns.lineplot(x='release_year', y='tmdb_score', data=df)
plt.title("Score Trend Over Time")
plt.show()

# 15. Pairplot (Overall Relationships)
sns.pairplot(df[['release_year','runtime','tmdb_popularity','tmdb_score']])
plt.show()