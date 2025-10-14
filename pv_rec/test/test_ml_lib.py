import pickle
import random
import shutil
from unittest import mock

import pandas as pd
import pytest
from more_itertools.more import side_effect
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers.models.cvt\
    .convert_cvt_original_pytorch_checkpoint_to_pytorch import \
    embeddings

import utils.data_factory
from utils import ml_lib

random.seed(42)


class TestProductShelf:
    @pytest.fixture
    def prod_embeddings(self):
        with open("data/test_ml_lib/encoded_products.pkl", "rb") as file:
            embeddings = pickle.load(file)
        return embeddings

    @pytest.fixture
    def product_shelf_obj(self, prod_embeddings):
        return utils.ml_lib.ProductShelf(prod_embeddings,
                                         clustering=KMeans(n_clusters=3)
                                         )

    def test_fit(self, product_shelf_obj):
        product_shelf_obj.fit()

        assert product_shelf_obj.pipeline

    def test_cluster_products(self, product_shelf_obj):
        product_shelf_obj.fit()
        product_shelf_obj.cluster_products()

        assert True

    def test_append_to_df(self, product_shelf_obj):
        data = pd.read_csv(
            r"data/test_data_factory/wlw_test_data.csv",
            index_col=0
        )
        wlw_pipeline = utils.data_factory.WlwPipeline(data)
        wlw_pipeline.transform()

        product_shelf_obj.fit()
        _ = product_shelf_obj.append_to_df(wlw_pipeline.data)
        assert True

    @pytest.mark.parametrize('products, expected',
                             [
                                 [[], 0],
                                 [['Stretchfolien'], 42]
                             ]

                             )
    def test_get_products(self, product_shelf_obj, products, expected):
        with mock.patch('sklearn.pipeline.Pipeline.predict',
                        return_value=42
                        ):
            obj_ut = product_shelf_obj.get_product_labels(products=products)

        assert int(obj_ut) == expected


class TestAbstractProductEncoder:
    @pytest.fixture
    def abstract_product_encoder_obj(self):
        return ml_lib.AbstractProductEncoder(products=['some Product1',
                                                       'some Product2',
                                                       'some Product3',
                                                       'some Product4'],
                                             model_name='random_model'
                                             )

    def test_embedd_it_api(self, abstract_product_encoder_obj, requests_mock):
        with requests_mock:
            embeddings = abstract_product_encoder_obj.embedd_it_api('product')

        # abstract_product_encoder_obj.set_llm_endpoint(
        #     model_id='intfloat/multilingual-e5-large')
        # embeddings = abstract_product_encoder_obj.embedd_it(
        #     ['Apfel',
        #      "Birne",
        #      "Banane",
        #      "Flugzeug",
        #      "Raumschiff"]
        # )

        assert (embeddings == pd.DataFrame([1, 2, 3])).all().all()

    def test_embedd_it_local(self,
                             abstract_product_encoder_obj,
                             sentence_transformers_mock):

        abstract_product_encoder_obj.model_name =\
            "Snowflake/snowflake-arctic-embed-l-v2.0"
        embeddings = abstract_product_encoder_obj.embedd_it_local(
            ['Apfel',
             "Birne",
             "Banane",
             "Flugzeug",
             "Raumschiff"]
        )

        assert (embeddings == [42]).all().all()

    def test_embedd_it_fail(self, abstract_product_encoder_obj):
        with mock.patch(
                'requests.post',
                return_value=mock.MagicMock(status_code=999)
        ):
            with pytest.raises(shutil.ExecError) as exception_info:
                abstract_product_encoder_obj.embedd_it_api('product')

        assert exception_info.typename == "ExecError"

    def test_embedd_it_parallel(self, abstract_product_encoder_obj,
                                requests_mock
                                ):
        with requests_mock, \
                mock.patch('utils.ml_lib.AbstractProductEncoder'
                           '._post_process_results', return_value=42
                           ):
            embeddings = \
                abstract_product_encoder_obj.embedd_it_parallel(
                    n_jobs=1,
                    n_chunks=2
                )

        assert embeddings == 42

    def test__join_results(self, abstract_product_encoder_obj):
        result1 = [1, 2, 3]
        result2 = [4, 5, 6]
        expected = pd.DataFrame([1, 2, 3, 4, 5, 6], index=(0, 1, 2, 0, 1, 2))

        obj_ut = abstract_product_encoder_obj._join_results([result1, result2])

        assert (obj_ut == expected).all().all()

    def test__post_process_results(self, abstract_product_encoder_obj):
        result1 = [1, 2, 3]
        result2 = [4, 5, 6]
        expected = pd.DataFrame([1, 2, 3, 4, 5, 6], index=(0, 1, 2, 3, 4, 5))

        abstract_product_encoder_obj.products = [0, 1, 2, 3, 4, 5]

        obj_ut = \
            abstract_product_encoder_obj._post_process_results(
                [result1, result2]
            )

        assert (obj_ut == expected).all().all()
