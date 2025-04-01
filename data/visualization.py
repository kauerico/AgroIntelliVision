import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_class_distribution(class_names, class_counts):
    plt.figure(figsize=(14,7))
    sns.barplot(x=class_names, y=class_counts)
    plt.title('Distribuição de Classes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()