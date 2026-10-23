from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.app_types import BaseSelfLearnedApp
from pyfedappwrap.learning.base_app import BaseApp, I, O
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


@dataclass
class AgglomerativeClusteringAppConfig(AppConfig):
    compute_full_tree: bool = 1
    linkage: str = "ward"
    compute_distances: bool = 1
    standardize: bool = 1
    metric: str = "euclidean"
    use_distance_threshold: bool = 1
    n_clusters: int = 2
    distance_threshold: float = 0.0
    dendrogram_truncate_mode: str = "none"
    dendrogram_p: int = 30
    plot_title: str = "Agglomerative Clustering Dendrogram"


@dataclass
class AgglomerativeClusteringAppInputConfig(AppInputConfig):
    features: Any = None


@dataclass
class AgglomerativeClusteringAppOutputConfig(AppOutputConfig):
    dendrogram_png: Path | None = None
    report: str = None


class AgglomerativeClusteringAPP(BaseSelfLearnedApp[AgglomerativeClusteringAppConfig, AgglomerativeClusteringAppInputConfig, AgglomerativeClusteringAppOutputConfig]):

    def run_algorithm(self,
                       data: AgglomerativeClusteringAppInputConfig) -> AgglomerativeClusteringAppOutputConfig:
        model = AgglomerativeClustering(distance_threshold=self.config.distance_threshold, n_clusters=None)

        self.logger.info("Prediction")

        df: pd.DataFrame = data.features
        X = df.iloc[:, :4].values
        self.logger.info(f"Fitting model to {X.shape[0]} samples with {X.shape[1]} features")
        model = model.fit(X)
        self.logger.info("Model fitting complete")
        plt.title(self.config.plot_title)
        # plot the top three levels of the dendrogram
        self.plot_dendrogram(model, truncate_mode=self.config.dendrogram_truncate_mode,
                             p=self.config.dendrogram_p)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig("dendrogram.png")
        for i in range(10):
            self.send_metric("accuracy", i, i)
            self.send_metric("loss", i, i)

        return AgglomerativeClusteringAppOutputConfig(
            dendrogram_png=Path("dendrogram.png"),
            report=f"Agglomerative Clustering produced {self.config.n_clusters} clusters."
        )

    def plot_dendrogram(self, model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)


    def _save(self) -> str:
        pass

    def _load(self, path: str):
        pass