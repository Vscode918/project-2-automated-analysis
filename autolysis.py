import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to read and analyze the dataset
def analyze_dataset(filename):
    # Load the dataset with specified encoding to handle special characters
    data = pd.read_csv(filename, encoding='ISO-8859-1')

    # Show basic information about the dataset
    info = data.info()

    # Generate summary statistics for numeric columns only
    numeric_data = data.select_dtypes(include='number')
    summary = numeric_data.describe()

    # Detect missing values
    missing_values = data.isnull().sum()

    # Create a correlation matrix for numeric columns only
    correlation_matrix = numeric_data.corr()

    # Generate visualizations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.close()

    # Example of a bar plot (for countries in the happiness report)
    plt.figure(figsize=(10, 6))
    top_countries = data.nlargest(10, 'Life Ladder')  # Use 'Life Ladder' as the happiness score column
    sns.barplot(x='Country name', y='Life Ladder', data=top_countries)
    plt.title("Top 10 Happiest Countries")
    plt.xticks(rotation=45)
    plt.savefig("top_countries.png")
    plt.close()

    return info, summary, missing_values

# Function to create a narrative for the analysis
def generate_narrative(info, summary, missing_values, filename):
    narrative = f"# Data Analysis of {filename}\n\n"
    narrative += "## Data Information\n"
    narrative += f"{info}\n\n"
    narrative += "## Summary Statistics\n"
    narrative += f"{summary}\n\n"
    narrative += "## Missing Values\n"
    narrative += f"{missing_values}\n\n"
    narrative += "## Insights\n"
    narrative += "The dataset provides a wealth of information on happiness across various countries. "
    narrative += "The correlation matrix visualizes how different factors contribute to overall happiness scores. "
    narrative += "The top 10 happiest countries have been plotted to highlight the global distribution of happiness.\n\n"

    # Save the narrative to README.md
    with open("README.md", "w") as file:
        file.write(narrative)

# Main function to execute the analysis
def main():
    # Define the CSV filename
    filename = 'dataset.csv'  # Ensure the file is uploaded in Colab or present in the current directory

    # Analyze the dataset
    info, summary, missing_values = analyze_dataset(filename)

    # Generate the markdown narrative
    generate_narrative(info, summary, missing_values, filename)

    print("Analysis complete! Check the generated README.md and PNG files.")

# Run the main function
if __name__ == "__main__":
    main()
