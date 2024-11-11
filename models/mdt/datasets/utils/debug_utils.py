import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class TSNEHelper:
    def __init__(self, features: np.ndarray):
        self.features = features
        self.tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=0)
        print(f'Running T-SNE fit for ({features.shape})... (default: first half is target; second half is source)')
        self.tsne.fit_transform(self.features)
        self.embedding = self.tsne.embedding_
        self.kl_dist = self.tsne.kl_divergence_
        print('T-SNE embedding shape:', self.embedding.shape)  # (12000, 2)

    def plot_tsne(self, fn='tmp1'):
        fig, ax = plt.subplots(1)
        tsne_result = self.embedding
        # y = np.concatenate([
        #     np.zeros((tsne_result.shape[0] // 2)),
        #     np.ones((tsne_result.shape[0] // 2))]
        # , axis=0).flatten()
        y_text = ['target'] * (tsne_result.shape[0] // 2) + ['source'] * (tsne_result.shape[0] // 2)
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y_text})
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=3, alpha=0.5)
        lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        ax.set_title(f"KL divergence: {self.kl_dist}")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        save_path = f'{fn}.png'
        plt.savefig(save_path, dpi=300)
        print(f'fig saved to {save_path}')
        plt.clf()
        plt.close()

    def plot_heatmap(self):
        embedding = self.embedding
        min_val = embedding.min()
        max_val = embedding.max()
        heat_map = np.zeros((100, 100), dtype=np.uint8)
        for pairs in embedding:
            px = int((pairs[0] - min_val) / (max_val - min_val) * 98)
            py = int((pairs[1] - min_val) / (max_val - min_val) * 98)
            heat_map[px][py] += 1
        heat_map = ((heat_map / heat_map.max()) * 15).astype(np.uint8)

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.tick_params(labelsize=35)

        import seaborn as sns
        sns.heatmap(heat_map, xticklabels=20, yticklabels=20, cbar=None)

        save_name = 'features_' + self.task_name[:-12] + '.jpg'
        plt.savefig(os.path.join(self.save_path, save_name))
        plt.clf()
        plt.close()

        self.heat_map = heat_map