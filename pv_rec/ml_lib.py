import os
from abc import ABC
from shutil import ExecError

import dotenv
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import torch

dotenv.load_dotenv(dotenv.find_dotenv())


class _HuggingFaceConnection(ABC):
    def __init__(self):
        self.header={
            "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
            }

        self.llm_endpoint=None


class AbstractProductEncoder(_HuggingFaceConnection):
    def __init__(
        self,
        products,
        model_name
        ):
        super().__init__()

        self.model = None
        self.products=products
        self.model_name=model_name

    def embedd_it_local(self, texts):
        print("embedding texts")
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True
            )
        embeddings_df = pd.DataFrame(
            embeddings,
            index=texts,
        )
        return embeddings_df


class WlwProductEncoder(AbstractProductEncoder):
    def __init__(
        self,
        products,
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        needs_preprocessing=True
        ):
        super().__init__(
            products=products,
            model_name=model_name
            )
        if needs_preprocessing:
            self._pre_process_data()

    def _pre_process_data(self):
        self._make_products_unique()
        self._filter_data_for_strings()

    def _make_products_unique(self):
        self.products=list(self.products.explode().unique())

    def _filter_data_for_strings(self):
        self.products=[iproduct_cat for iproduct_cat in
                       self.products if
                       not isinstance(iproduct_cat, float)]


class ProductShelf:
    def __init__(
        self,
        clustering=None,
        scaler=RobustScaler(with_centering=False,
                            with_scaling=True,
                            quantile_range=(10, 90)
                            )
        ):
        self.clustering=clustering
        self.scaler=scaler

        self.pipeline=self._set_pipeline()

        self.product_shelves=None

    def cluster_products(self, embeddings):
        self.product_shelves= \
            self.pipeline.predict(embeddings)

        return self.product_shelves

    def _set_pipeline(self):
        return Pipeline([
            ("scaler", self.scaler),
            ("clustering", self.clustering)
            ]
            )

    def fit(self, embeddings):
        self.pipeline.fit(embeddings)

    def append_to_df(self, data, embeddings):
        labeled_products=pd.DataFrame()
        for icompany, idata in data.iterrows():
            iproducts=list(idata.product_categories)
            if iproducts==[]:
                continue
            ilabels=self.get_product_labels(iproducts, embeddings)

            count_categories=np.unique(ilabels, return_counts=True)
            labeled_products=pd.concat(
                [labeled_products,
                 pd.DataFrame(index=[icompany],
                              columns=count_categories[0],
                              data=count_categories[1].reshape(1, -1)
                              )]
                )

        self._change_column_names(labeled_products)
        labeled_products.fillna(0, inplace=True)
        return pd.concat([data, labeled_products], axis=1)

    @staticmethod
    def _change_column_names(labeled_products):
        labeled_products.columns=['product_label_' + str(int(icolumns)) for
                                  icolumns in labeled_products.columns]

    def get_product_labels(self, products, embeddings):
        if not products:
            return np.ndarray([])
        return self.pipeline.predict(embeddings.loc[products, :])
