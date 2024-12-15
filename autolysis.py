import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate

load_dotenv()

if "AIPROXY_TOKEN" not in os.environ:
    raise EnvironmentError("The environment variable 'AIPROXY_TOKEN' must be set.")

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = os.environ["AIPROXY_TOKEN"]

def analyze_and_generate_report(csv_filename):
    try:
        df = pd.read_csv(csv_filename, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    print("Dataset Info:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(f"\nNumeric Columns: {list(numeric_cols)}")

    summary_stats = df[numeric_cols].describe()
    print("\nSummary Statistics:")
    print(summary_stats)

    # Normality Test
    normality_results = {}
    for col in numeric_cols:
        stat, p_value = shapiro(df[col].dropna())
        normality_results[col] = {
            "Statistic": stat,
            "P-Value": p_value,
            "Normal": p_value > 0.05
        }

    print("\nNormality Test Results:")
    print(pd.DataFrame(normality_results).transpose())

    # Correlation Matrix
    heatmap_file = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        heatmap_file = "correlation_heatmap.png"
        plt.savefig(heatmap_file)
        plt.close()
        print(f"Saved correlation heatmap as {heatmap_file}")

    # Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = numeric_cols
    vif_data['VIF'] = [variance_inflation_factor(df[numeric_cols].dropna().values, i) for i in range(len(numeric_cols))]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)

    # Histograms
    histograms_file = "histograms.png"
    plt.figure(figsize=(15, len(numeric_cols) * 4))
    for i, col in enumerate(numeric_cols):
        plt.subplot(len(numeric_cols), 1, i + 1)
        sns.histplot(df[col].dropna(), kde=True, bins=20, color='blue')
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(histograms_file)
    plt.close()
    print(f"Saved histograms as {histograms_file}")

    # Clustering with KMeans
    cluster_file = None
    if len(numeric_cols) > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols].dropna())
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)
        cluster_df = pd.DataFrame(reduced_data, columns=["PCA1", "PCA2"])
        cluster_df["Cluster"] = clusters

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=cluster_df, palette="viridis")
        plt.title("Cluster Analysis (PCA Reduced)")
        cluster_file = "cluster_analysis.png"
        plt.savefig(cluster_file)
        plt.close()
        print(f"Saved cluster analysis plot as {cluster_file}")

    # Prepare Data for LLM
    dataset_overview = {
        "Number of rows": df.shape[0],
        "Number of columns": df.shape[1],
        "Numeric Columns": list(numeric_cols),
    }

    summary_stats_table = tabulate(summary_stats, headers='keys', tablefmt='pipe')
    
    # Prepare LLM prompt
    narrative = generate_llm_narrative(dataset_overview, summary_stats_table, heatmap_file, normality_results)

    # Generate Markdown report
    markdown_content = f"""
# Analysis Report

## Dataset Overview
- Number of rows: {df.shape[0]}
- Number of columns: {df.shape[1]}

## Numeric Columns
{list(numeric_cols)}

## Summary Statistics
{summary_stats_table}

## Normality Test Results
{pd.DataFrame(normality_results).transpose().to_markdown()}

"""

    if heatmap_file:
        markdown_content += f"## Correlation Matrix\n![Correlation Matrix Heatmap]({heatmap_file})\n"

    markdown_content += f"## Histograms\n![Histograms]({histograms_file})\n"

    if cluster_file:
        markdown_content += f"## Cluster Analysis\n![Cluster Analysis]({cluster_file})\n"

    markdown_content += f"\n## Narrative Analysis\n{narrative}"

    with open("README.md", "w") as md_file:
        md_file.write(markdown_content)
    print("Generated README.md report.")

def generate_llm_narrative(overview, summary_stats, heatmap_file, normality_results):
    try:
        condensed_normality = {k: v['Normal'] for k, v in normality_results.items()}
        prompt = (
            "You are a data analyst. Write a detailed narrative based on the analysis:\n\n"
            f"### Dataset Overview\n{overview}\n\n"
            f"### Summary Statistics\n{summary_stats}\n\n"
            f"### Normality Tests\n{condensed_normality}\n\n"
        )

        if heatmap_file:
            prompt += (
                "### Correlation Matrix\n"
                "A correlation heatmap was generated to analyze relationships between numeric variables."
            )

        prompt += "\nDiscuss key findings, implications, and actionable recommendations."

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating narrative: {e}")
        return "An error occurred while generating the narrative."

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_filename>")
    else:
        csv_filename = sys.argv[1]
        analyze_and_generate_report(csv_filename)
