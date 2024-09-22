from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np

def plot_permutation_importance(model, X, y, feature_names, scoring, n_repeats=10, random_state=42):
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state)

    importance_means = result.importances_mean
    importance_std = result.importances_std
    indices = np.argsort(importance_means)[::-1]  # Sort in descending order

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importance_means[indices], xerr=importance_std[indices], align='center', color='skyblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Mean decrease in custom score")
    plt.title("Permutation Importance using Custom Scorer")
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.tight_layout()
    plt.show()