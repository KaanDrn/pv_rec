from abc import ABCMeta

from pv_rec import data_classes
from pv_rec.ml_lib import WlwProductEncoder

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

class WlwDataUtility(metaclass=ABCMeta):
    @staticmethod
    def _get_fill_values():
        fill_nan_values={
            "company_street": "unknown",
            "company_zip": 0,
            "company_city": "unknown",
            "distribution-area": "unknown",
            "employee-count": "unknown",
            "founding-year": 0,
            "Dienstleister": False,
            "Großhändler": False,
            "Hersteller/Fabrikant": False,
            "Lieferant": False,

            }
        return fill_nan_values

    @staticmethod
    def fill_nan_values(data):
        data.fillna(value=WlwDataUtility._get_fill_values(),
                    inplace=True
                    )

    @staticmethod
    def check_index(data):
        if 'company_name' in data.columns:
            data.set_index('company_name', inplace=True)

    @staticmethod
    def transform_address(data):
        data['company_address']=data.company_street + ', ' + \
                                data.company_zip.astype(str) + ' ' + \
                                data.company_city
        data['company_housenumber']= \
            [istreet.split(' ')[-1] for istreet in data.company_street]

    @staticmethod
    def parse_data(data):
        # This method does not actually change the data, but the dataclass
        # helps to understand what the data should look like
        return data_classes.parse_wlw_data(data)

    @staticmethod
    def drop_missing_address(data):
        data.dropna(subset=['company_street',
                            'company_zip',
                            'company_city'],
                    axis=0,
                    inplace=True
                    )

    @staticmethod
    def resort_columns(data):
        return data.reindex(sorted(data.columns), axis=1)

    @staticmethod
    def preprocess_data(data):
        # These Methods modify the data inplace, hence a reassignment to data
        # is not needed
        WlwDataUtility.check_index(data)
        WlwDataUtility.drop_missing_address(data)
        WlwDataUtility.fill_nan_values(data)

        return WlwDataUtility.parse_data(data)


class WlwPipeline(WlwDataUtility):
    def __init__(self, data):
        self.data=data

    def transform(self):
        self.data=self.preprocess_data(self.data)
        # self.transform_address(self.data)
        self.data=WlwDataUtility.resort_columns(self.data)

    def encode_products(
        self, n_jobs=-1, n_chunks=4,
        model_name="intfloat/multilingual-e5-large"
        ):
        encoder=WlwProductEncoder(self.data.product_categories,
                                  model_name=model_name
                                  )

        embeddings=encoder.embedd_it_parallel(n_jobs=n_jobs,
                                              n_chunks=n_chunks
                                              )
        return embeddings


class DataMaster:
    def __init__(self,
                 mastr_filepath: str,
                 solar_filepath: str,
                 wlw_filepath: str):
        self.mastr_data = self.load_mastr_data(filepath=mastr_filepath)
        self.wlw_data = self.load_wlw_data(filepath=wlw_filepath,
                                           solar_filepath=solar_filepath)
        self.test_data = None

    # %% Mastr Data
    @staticmethod
    def load_mastr_data(filepath: str):
        mastr_data=pd.read_csv(filepath, index_col=0)

        data_pipeline=WlwPipeline(mastr_data)
        data_pipeline.transform()
        mastr_data=data_pipeline.data

        mastr_data=DataMaster.drop_duplicates(mastr_data)

        # resort columns
        mastr_data=mastr_data.reindex(sorted(mastr_data.columns), axis=1)

        return mastr_data

    @staticmethod
    def drop_duplicates(mastr_data: pd.DataFrame) -> pd.DataFrame:
        """
        Drop duplicates in mastr data by aggregating them

        Parameters
        ----------
        mastr_data : pd.DataFrame
            Dataframe with mastr data

        Returns
        -------
        pd.DataFrame
            Dataframe with aggregated mastr data

        """
        agg_dict= \
            {
                'distribution_area': 'max',
                'employee_count': 'max',
                'founding_year': 'min',
                'installed_power': 'mean',
                'is_producer': 'max',
                'is_sales': 'max',
                'is_serviceprovider': 'max',
                'is_wholesales': 'max',
                'num_modules': 'mean',
                'product_categories': 'first',
                }
        mastr_data=mastr_data.groupby(level=0).agg(agg_dict)
        return mastr_data

    # %% Solar Data
    @staticmethod
    def load_solar_data(filepath: str):
        solar_wlw=pd.read_csv(filepath, index_col=0)
        solar_wlw.drop(["CO2_19_5", "STR_19_5"], axis=1, inplace=True)

        DataMaster.apply_naming_convention(solar_wlw)
        DataMaster.apply_error_placeholder(solar_wlw)
        DataMaster.apply_correct_datatypes(solar_wlw)
        return solar_wlw

    @staticmethod
    def apply_correct_datatypes(solar_wlw):
        module=[]
        leistung=[]
        for iindex, idata in solar_wlw.iterrows():
            try:
                module.append(int(idata["Anzahl Module"].replace(".0", "")))
            except AttributeError:
                module.append(0)
            leistung.append(float(idata["Leistung"]))
        solar_wlw["Anzahl Module"]=module
        solar_wlw["Leistung"]=leistung

    @staticmethod
    def apply_error_placeholder(solar_wlw):
        solar_data_map={
            "no close roof found": 0,
            "'NoneType' object has no attribute 'latitude'": 0,
            "more than 1 roof object found": 0
            }

        # happens inplace so no need to return or reassign
        solar_wlw['Anzahl Module']=solar_wlw['Anzahl Module']. \
            replace(solar_data_map)
        solar_wlw['Leistung']=solar_wlw['Leistung']. \
            replace(solar_data_map)

    @staticmethod
    def apply_naming_convention(solar_wlw):
        column_map={
            "KW_19_5": "Leistung",
            "MODANETTO": "Anzahl Module"
            }
        # happens inplace so no need to return or reassign
        solar_wlw.rename(columns=column_map, inplace=True)

    # %% WLW Data
    def load_wlw_data(self, filepath, solar_filepath):
        solar_data = self.load_solar_data(solar_filepath)

        wlw_data = pd.read_csv(filepath, index_col=0)

        wlw_data = self.merge_solar_and_wlw(solar_data, wlw_data)

        data_pipeline=WlwPipeline(wlw_data)
        data_pipeline.transform()
        wlw_data=data_pipeline.data

        return wlw_data

    def merge_solar_and_wlw(self, solar_data, wlw_data):
        merged_data = wlw_data.join(solar_data)
        return merged_data

    # %% Test Data
    def extract_test_data(self, data, sample_size=100, random_state=42):
        self.test_data=data.sample(sample_size, random_state=random_state)
        data.drop(self.test_data.index, inplace=True)
