import numpy as np
import torch
from sklearn.metrics import pairwise
import plotly.express as px
import seaborn as sns

class KernelKMeans:
    def __init__(self, n_clusters=8, max_iter=300, kernel=pairwise.linear_kernel):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.kernel = kernel

    def _initialize_clusters(self, X):
        self.num_samples = X.shape[0]
        self.labels = torch.randint(low=0, high=self.n_clusters, size=(self.num_samples,))
        self.kernel_matrix = torch.tensor(self.kernel(X))

    def fit_predict(self, X):
        self._initialize_clusters(X)

        for _ in range(self.max_iter):
            obj = torch.tile(torch.diag(self.kernel_matrix).reshape((-1, 1)), self.n_clusters)
            samples_in_clusters = [self.kernel_matrix[:, self.labels == c] for c in range(self.n_clusters)]

            for c in range(self.n_clusters):
                obj[:, c] -= 2 * torch.sum(samples_in_clusters[c], dim=1) / len(samples_in_clusters[c][0])
                obj[:, c] += torch.sum(self.kernel_matrix[self.labels == c][:, self.labels == c]) / (len(samples_in_clusters[c]) ** 2)

            self.labels = torch.argmin(obj, dim=1)

        return self.labels.numpy()

def visualize_clusters(X, labels):
    data = {'Feature 1': X[:, 0], 'Feature 2': X[:, 1], 'Cluster': labels}
    df = pd.DataFrame(data)
    
    # Plotly Scatter Plot
    fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Cluster', title='Kernel K-Means Clustering')
    fig.show()

    # Seaborn Scatter Plot
    sns.scatterplot(x='Feature 1', y='Feature 2', hue='Cluster', data=df)
    plt.title('Kernel K-Means Clustering')
    plt.show()

# Example usage
# X = your_data
# kkm = KernelKMeans(n_clusters=2, max_iter=100, kernel=pairwise.rbf_kernel)
# labels = kkm.fit_predict(X)

# Visualize the result
# visualize_clusters(X, labels)
