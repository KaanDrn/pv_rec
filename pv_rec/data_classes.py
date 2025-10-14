import ast
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class DataclassWlwData:
    #company_street: str
    #company_zip: int
    #company_city: str
    distribution_area: str
    founding_year: int
    employee_count: str
    product_categories: set[str]
    is_producer: bool = False
    is_serviceprovider: bool = False
    is_wholesales: bool = False
    is_sales: bool = False
    num_modules: int = None
    installed_power: float = None
    # latitude: float = None
    # longitude: float = None


def parse_wlw_data(data, **kwargs):
    parsed_data = DataclassWlwData(
        #company_street=data.company_street,
        #company_zip=data.company_zip.astype(int),
        #company_city=data.company_city,
        distribution_area=data["distribution-area"],
        founding_year=data["founding-year"].astype(int),
        employee_count=data["employee-count"],
        product_categories=data["product_categories"].apply(ast.literal_eval),
        is_producer=data['Hersteller/Fabrikant'],
        is_serviceprovider=data['Dienstleister'],
        is_wholesales=data['Großhändler'],
        is_sales=data["Lieferant"],
        num_modules=data["Anzahl Module"],
        installed_power=data["Leistung"],
        # latitude=data["Breitengrad"],
        # longitude=data["Längengrad"]
    )
    return pd.DataFrame.from_dict(asdict(parsed_data))
