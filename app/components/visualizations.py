import matplotlib.pyplot as plt
import seaborn as sns


# Plot how many students are in each risk level (Low, Medium, High)
def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df,
        x="Mental_Health_Risk",
        order=["Low", "Medium", "High"],
        palette=["#2ecc71", "#f1c40f", "#e74c3c"],
        ax=ax,
    )
    ax.set_title("Student Mental Health Risk Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Students")
    return fig


# Plot histogram for any numerical column
def plot_feature_distribution(df, column):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=column, hue="Mental_Health_Risk", kde=True, ax=ax)
    ax.set_title(f"Distribution of {column} by Risk Level")
    return fig


# Plot correlation heatmap for numerical features
def plot_correlation_heatmap(df):
    df_corr = df.copy()
    df_corr["Risk_Score"] = df_corr["Mental_Health_Risk"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    )
    num_df = df_corr.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


# Plot confusion matrix for model evaluation
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig
