import matplotlib.pyplot as plt


def save_bar_chart(df, x_col, y_col, title, output_path, xlabel="", ylabel="Bias rate"):
    plt.figure(figsize=(10, 5))
    plt.bar(df[x_col].astype(str), df[y_col].astype(float))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
