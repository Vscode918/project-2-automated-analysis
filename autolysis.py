# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "openai",
#     "scikit-learn",
#     "tabulate",
#     "fastapi",
#     "uvicorn",
#     "seaborn",
# ]
# ///
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tabulate import tabulate

load_dotenv()
# Set the AI Proxy token
if "AIPROXY_TOKEN" not in os.environ:
    raise EnvironmentError("The environment variable 'AIPROXY_TOKEN' must be set.")
    
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = os.environ["AIPROXY_TOKEN"]

def analyze_and_generate_report(csv_filename):
    # Load the dataset
    try:
        df = pd.read_csv(csv_filename, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Display basic dataset info
    print("Dataset Info:")
    print(df.info())
    print("\nSample Data:")
    print(df.head())

    # Handle numeric columns (conversion and cleaning)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    print(f"\nNumeric Columns: {list(numeric_cols)}")

    # Compute summary statistics
    summary_stats = df[numeric_cols].describe()
    print("\nSummary Statistics:")
    print(summary_stats)

    # Create correlation matrix
    heatmap_file = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()

        # Save the correlation matrix heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        heatmap_file = "correlation_heatmap.png"
        plt.savefig(heatmap_file)
        plt.close()
        print(f"Saved correlation heatmap as {heatmap_file}")
    else:
        print("Not enough numeric columns for correlation analysis.")

    # Histograms for numeric columns
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

    # Cluster analysis visualization
    cluster_file = None
    if len(numeric_cols) > 1:
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_data = df[numeric_cols].dropna()
        clusters = kmeans.fit_predict(cluster_data)

        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(cluster_data)
        cluster_df = pd.DataFrame(reduced_data, columns=["PCA1", "PCA2"])
        cluster_df["Cluster"] = clusters

        # Scatter plot of clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=cluster_df, palette="viridis")
        plt.title("Cluster Analysis (PCA Reduced)")
        cluster_file = "cluster_analysis.png"
        plt.savefig(cluster_file)
        plt.close()
        print(f"Saved cluster analysis plot as {cluster_file}")

    # Prepare data for LLM narrative
    dataset_overview = {
        "Number of rows": df.shape[0],
        "Number of columns": df.shape[1],
        "Numeric Columns": list(numeric_cols),
    }

    summary_stats_table = tabulate(summary_stats, headers='keys', tablefmt='pipe')

    # Generate LLM-based narrative
    narrative = generate_llm_narrative(dataset_overview, summary_stats_table, heatmap_file)

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

"""

    if heatmap_file:
        markdown_content += "## Correlation Matrix\n"
        markdown_content += f"![Correlation Matrix Heatmap]({heatmap_file})\n"

    markdown_content += f"## Histograms\n![Histograms]({histograms_file})\n"

    if cluster_file:
        markdown_content += f"## Cluster Analysis\n![Cluster Analysis]({cluster_file})\n"

    markdown_content += "\n## Narrative Analysis\n"
    markdown_content += narrative

    # Save Markdown report
    with open("README.md", "w") as md_file:
        md_file.write(markdown_content)
    print("Generated README.md report.")

def generate_llm_narrative(overview, summary_stats, heatmap_file):
    try:
        # Prepare the prompt for the LLM
        prompt = (
            "You are a data analyst. Write a clear, insightful narrative based on the following analysis:\n\n"
            f"### Dataset Overview\n{overview}\n\n"
            f"### Summary Statistics\n{summary_stats}\n\n"
        )

        if heatmap_file:
            prompt += (
                "### Correlation Matrix\n"
                "A correlation heatmap was generated. Mention its importance in finding relationships between numeric variables.\n"
            )

        prompt += "\nDiscuss key findings, potential implications, and recommendations based on this data."

        # Call the LLM
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        # Extract the response text
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating narrative: {e}")
        import traceback
        traceback.print_exc()
        return "An error occurred while generating the narrative."

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <csv_filename>")
    else:
        csv_filename = sys.argv[1]
        analyze_and_generate_report(csv_filename)
