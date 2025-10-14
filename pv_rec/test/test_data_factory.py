from unittest import mock

import pandas as pd
import pytest

from utils import data_factory as factory


class TestWlwPipeline:
    @pytest.fixture
    def wlw_test_data(self):
        with open('data/test_data_factory/wlw_test_data.csv',
                  encoding='utf-8') as f:
            wlw_data = pd.read_csv(f, index_col=0)

        return wlw_data

    @pytest.fixture()
    def wlw_pipeline_obj(self, wlw_test_data):
        obj = factory.WlwPipeline(wlw_test_data)
        return obj

    def test_transform(self, wlw_pipeline_obj):
        # debug
        mastr_data = pd.read_csv(
            '/Users/kaanduran/Documents/repos/phd_CSPV_industry/reports/6_merge_wlw_and_mastr/input/niedersachsen_stromerzeuger_mastr_to_wlw_style.csv',
            index_col=0,
        )

        data_pipeline = factory.WlwPipeline(mastr_data)
        data_pipeline.transform()
        # debug end

        wlw_pipeline_obj.transform()
        obj_ut = wlw_pipeline_obj.data

        assert "company_address" in obj_ut.columns
        assert "company_housenumber" in obj_ut.columns

    def test_encode_products(self, wlw_pipeline_obj):
        wlw_pipeline_obj.transform()
        with mock.patch(
                'utils.ml_lib.AbstractProductEncoder.embedd_it_parallel',
                return_value=True):
            obj_ut = wlw_pipeline_obj.encode_products(
                n_jobs=-1,
                n_chunks=2)
        assert obj_ut
