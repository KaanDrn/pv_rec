import os
import random
import time
import urllib

import matplotlib
import numpy as np
import pandas as pd
import requests
import textdistance
from bs4 import BeautifulSoup
from geopy import Nominatim
from matplotlib import pyplot as plt
from pyproj import Transformer

matplotlib.use('TkAgg')


class FirmenDbCrawler:
    def __init__(self, web_url):
        self.search_url = web_url
        self.base_url = 'http://firmendb.de'

        self.company_meta = {}

    def crawl_firmen_db(self):
        company_data = []
        response = requests.get(self.search_url)

        soup = BeautifulSoup(response.text,
                             features="lxml"
                             )
        company_soup = soup.find_all('li', 'list-group-item')

        for ientries in company_soup:
            try:
                company_url = self._get_company_url(ientries)
            except AttributeError:
                # if you encounter a google ad
                continue
            company_website = BeautifulSoup(requests.get(company_url).text,
                                            features="lxml"
                                            )

            address_box = self._get_address_box(company_website)
            address_box_info = self._extract_address_box(address_box)

            company_info_box = self.get_info_box(company_website)
            company_info = self.get_company_info(company_info_box)

            company_data.append(address_box_info | company_info)
        return pd.DataFrame(company_data)

    def get_company_info(self, company_info_box):
        info_keys, info_values = self.get_key_value_pairs(company_info_box)

        company_info = {}
        for ikey, ivalue in zip(info_keys, info_values):
            company_info[ikey.text[:-1]] = ivalue.text

        self._clean_data(company_info)
        return company_info

    @staticmethod
    def _clean_data(company_info):
        try:
            del company_info['Ofizieller Name']
        except:  # noqa: E722
            pass

        try:
            company_info['Branche'], company_info['Branche_Hauptbranche'] = \
                company_info.get('Branche').split(' / ')
        except:  # noqa: E722
            pass
        try:
            company_info['Mitarbeiter'] = int(company_info.
                                              get('Mitarbeiter').
                                              split(' ')[0].
                                              replace('.', '')
                                              )
        except:  # noqa: E722
            pass
        try:
            company_info['Firmengründung'] = int(company_info.
                                                 get('Firmengründung')
                                                 )
        except:  # noqa: E722
            pass
        try:
            company_info['Stammkapital'] = int(company_info.
                                               get('Stammkapital').
                                               split(' ')[0].
                                               replace('.', '')
                                               )
        except:  # noqa: E722
            pass
        try:
            company_info['Umsatz'] = int(company_info.
                                         get('Umsatz').
                                         split(' ')[0].
                                         replace('.', '')
                                         )
        except:  # noqa: E722
            pass
        return

    @staticmethod
    def get_key_value_pairs(company_info_box):
        info_keys = company_info_box.find_all('dt')
        info_values = company_info_box.find_all('dd')
        return info_keys, info_values

    @staticmethod
    def get_info_box(company_website):
        company_info_box = company_website.find(
            'dl',
            'dl-horizontal dl-antiblock'
        )
        return company_info_box

    def _get_company_url(self, website_soup):
        local_url = website_soup.find('a').get('href')
        company_url = self.base_url + local_url[2:]
        return company_url

    @staticmethod
    def _get_address_box(company_website):
        address_box = \
            company_website.find(
                'dl',
                'dl-horizontal dl-short dl-antiblock nomargin-bottom'
            )
        return address_box

    @staticmethod
    def _extract_address_box(address_box):
        address_box_collector = {}

        try:
            address_box_collector["company_name"] = address_box.find('span', {
                'itemprop': 'name'
            }
                                                                     ).text
        except AttributeError:
            pass
        try:
            address_box_collector["company_street"] = address_box.find(
                'span', {
                    'itemprop': 'streetAddress'
                }
            ).text
        except AttributeError:
            pass
        try:
            address_box_collector["company_zip"] = int(
                address_box.find('span', {
                    'itemprop': 'postalCode'
                }
                                 ).text
            )
        except AttributeError:
            pass
        try:
            address_box_collector["company_city"] = address_box.find('span', {
                'itemprop': 'addressLocality'
            }
                                                                     ).text
        except AttributeError:
            pass
        try:
            address_box_collector["company_tel"] = address_box.find('dd', {
                'itemprop': 'telephone'
            }
                                                                    ).text
        except AttributeError:
            pass
        try:
            address_box_collector["_company_website"] = address_box. \
                find_all('dd')[-1].get_text('\"')
        except AttributeError:
            pass

        return address_box_collector


class SolarCatastreCrawler:
    def __init__(self):
        self.searcher = Nominatim(user_agent='solar_address_search')

        self.coordinates = None
        self.solar_query = None

        self.RELEVANT_FIELDS = ('STR_19_5', 'CO2_19_5', 'KW_19_5',
                                'MODANETTO', 'EIGNGPVI', 'DACHTYP', 'BELEGT_0')

        # ToDo: this are all relevant fields, but we will try with only three
        #  at the moment
        # self.RELEVANT_FIELDS = ('OBJECTID', 'ID', 'GEB_ID', 'PRZ_MEAN',
        #                         'RICHTUNG', 'NEIGUNG', 'MOD_ID',
        #                         'WIN_MEDIAN',
        #                         'YR_MEDIAN', 'JAN_MEDIAN', 'FEB_MEDIAN',
        #                         'MRZ_MEDIAN', 'APR_MEDIAN', 'MAI_MEDIAN',
        #                         'JUN_MEDIAN', 'JUL_MEDIAN', 'AUG_MEDIAN',
        #                         'SEP_MEDIAN', 'OKT_MEDIAN', 'NOV_MEDIAN',
        #                         'DEZ_MEDIAN', 'DACHTYP', 'AREA_2D',
        #                         'MODAREA',
        #                         'RADABS', 'GRADPRZ', 'STR_19_5', 'CO2_19_5',
        #                         'KW_19_5', 'KWH_KWP', 'EIGNGPVI', 'ANZAHL_0',
        #                         'BELEGT_0', 'MODANETTO')

    def crawl_solar_cadastre(self, address):
        try:
            self.find_address(address)
        except AttributeError as e:
            print('Address not found')
            return pd.Series(data=[e] * len(self.RELEVANT_FIELDS),
                             index=self.RELEVANT_FIELDS
                             ).drop('EIGNGPVI')
        try:
            data = self.get_closest_roof_data()
        except ValueError as e:
            print(e)
            return pd.Series(data=[e] * len(self.RELEVANT_FIELDS),
                             index=self.RELEVANT_FIELDS
                             ).drop('EIGNGPVI')
        data = self.get_roof_data_from_id(building_id=data.get('GEB_ID'))
        data = self.aggreagate_data(data)

        return data

    def _set_solar_query(
            self, xmax, xmin, ymax, ymin,
            out_fields, where
    ):
        out_fields_query_string = ''
        for iout_fields in out_fields:
            out_fields_query_string += iout_fields + '%2C'
        out_fields_query_string = out_fields_query_string[:-3]

        self.solar_query = f'https://gis-services.landkreishildesheim.de' \
                           f'/arcgis/rest/services/Solar/' \
                           f'Solarkataster_Vektor_Photovoltaik/MapServer/0/' \
                           f'query?f=json&returnGeometry=true&spatialRel' \
                           f'=esriSpatialRelIntersects' \
                           f'&geometry=%7B%22' \
                           f'xmin%22%3A{xmin}%2C%22' \
                           f'ymin%22%3A{ymin}%2C%22' \
                           f'xmax%22%3A{xmax}%2C%22' \
                           f'ymax%22%3A{ymax}%2C%22spatialReference%22' \
                           f'%3A%7B%22wkid%22%3A102100%7D%7D&geometryType' \
                           f'=esriGeometryEnvelope&inSR=102100&outFields' \
                           f'={out_fields_query_string}&outSR=102100'

        if where is not None:
            where_query_string = '&where='
            for ikey, ivalue in where.items():
                where_query_string += f'{ikey}%3D%27{ivalue}%27&'
            where_query_string = where_query_string[:-1]
            self.solar_query += where_query_string

    def find_address(self, address):
        # ToDo: The accuracy of this method should be investigated
        address_data = self.searcher.geocode(address)
        # ToDo: Maybe change to arcgis request url
        #  https://developers.arcgis.com/rest/geocode/api-reference
        #  /geocoding-find-address-candidates.htm
        #  #ESRI_SECTION1_15856BE1AD294298954B2E52172EE61B
        self.coordinates = (address_data.latitude, address_data.longitude)
        self._transform_coordinates()
        return

    def get_roof_data(self):
        response = requests.get(self.solar_query).json()

        return response.get('features')

    def set_request_area(
            self, offset=20.0, out_fields: set = ('*'), where=None
    ):

        self._set_solar_query(
            xmax=self.coordinates[0] + offset,
            xmin=self.coordinates[0] - offset,
            ymax=self.coordinates[1] + offset,
            ymin=self.coordinates[1] - offset,
            out_fields=out_fields,
            where=where
        )

    def _transform_coordinates(self):
        source_crs = "EPSG:4326"  # WGS 84
        target_crs = "EPSG:3857"  # Web Mercator

        transformer = Transformer.from_crs(source_crs, target_crs)
        self.coordinates = transformer.transform(
            xx=self.coordinates[0],
            yy=self.coordinates[1]
        )

    def get_closest_roof_data(self):
        self.set_request_area(offset=0.5, out_fields=['GEB_ID'])
        data = self.get_roof_data()
        if len(data) < 1:
            raise ValueError('no close roof found')

        if len(data) > 1:
            # transforming this into a set, because a set only contains
            # unique values
            all_ids = {idata['attributes'].get('GEB_ID') for idata in data}
            if len(all_ids) > 1:
                raise ValueError('more than 1 roof object found')

        return data[0].get('attributes')

    def get_roof_data_from_id(self, building_id, search_range=1000):
        """
        Parameters
        ----------
        building_id:
            GEB_ID of building you want to get the data from

        search_range:
            Range in which the GEB_ID should be searched in.
            search_area = search_center +- search_range in both directions
            (X and Y)

        Returns
        -------
        data:
            Pandas df from data avialable for a specific roof
        """
        self.set_request_area(offset=search_range,
                              out_fields=self.RELEVANT_FIELDS,
                              where={'GEB_ID': building_id}
                              )
        data = self.get_roof_data()
        roof_data = {}
        for inumber_roof, iroof in enumerate(data):
            roof_data[inumber_roof] = iroof.get('attributes')

        return pd.DataFrame(roof_data)

    def plot_geometries(self, data):
        geometries = data[0].get('geometry').get('rings')
        for igeometry in geometries:
            temp = pd.DataFrame(igeometry)
            plt.plot(temp[0], temp[1])
        return

    def aggreagate_data(self, data):
        # filter roofs, that are not suitable
        data = data.loc[:, list(data.loc['EIGNGPVI'].astype(bool))]
        data.drop('EIGNGPVI', inplace=True)

        data = data.sum(axis=1)
        return data


class WlwCrawler:
    def __init__(self, city, start_page=None, persisted_data_path=None):
        self.location = self.search_location(city=city)
        self.search_url = self.set_search_url(start_page)
        self.root_website = 'https://www.wlw.de'
        # is beeing filled in the crawling process. Is needed for further
        # improve crawling depth
        self._company_website = None
        self.data = self.get_persisted_data(
            data_path=persisted_data_path
        )

    def set_search_url(self, start_page):
        if start_page is None:
            base_url = \
                f"https://www.wlw.de/de/suche/?locationCountryCode=DE" \
                f"&locationKind=other&locationLatitude=" \
                f"{self.location.latitude}&" \
                f"locationLongitude={self.location.longitude}&" \
                f"locationName=" \
                f"{self.location.raw.get('name')}%2C%20Deutschland" \
                f"&locationRadius=50km&q=%27&sort=distance"
        else:
            base_url = \
                f"https://www.wlw.de/de/suche/page/" \
                f"{start_page}?locationCountryCode=DE" \
                f"&locationKind=other&locationLatitude=" \
                f"{self.location.latitude}&" \
                f"locationLongitude={self.location.longitude}&" \
                f"locationName=" \
                f"{self.location.raw.get('name')}%2C%20Deutschland" \
                f"&locationRadius=50km&q=%27&sort=distance"
        return base_url

    @staticmethod
    def search_location(city):
        searcher = Nominatim(user_agent='city_searcher')
        location = searcher.geocode(city)
        return location

    def next_page(self):
        print('\ngoing to the next page')
        if self.search_url.find('page') == -1:
            self.search_url = self.search_url[:28] + 'page/2' + \
                              self.search_url[28:]
        else:
            url_first, url_second = self.search_url.split('?')
            current_page_number = int(url_first[url_first.rfind('/') + 1:])
            url_first = url_first[:url_first.rfind('/') + 1]

            # remerge urls
            self.search_url = \
                url_first + \
                str(current_page_number + 1) + \
                '?' + \
                url_second

    def crawl_wlw_data(self):
        data = {}
        t0 = time.perf_counter()

        # ToDo: You can modify this to make the loop exactly as long as the
        #  number of pages should be, instead of adding it manually
        for ipage in range(0, 25):
            soup = self.get_soup()
            websites = self.get_company_websites(soup)
            # ToDo: Good coding practice? I dont think so!
            #  I am using an attribute as an iterator is this cool?
            for self._company_website in websites:
                content = requests.get(self._company_website)
                isoup = BeautifulSoup(content.text, features="lxml")

                company_info = self.extract_company_info(isoup)
                data.update({company_info['company_name']: company_info})

                self.random_sleep()

                t1 = time.perf_counter()
                print('elapse time: %.2f' % (t1 - t0))

            self.data = self.data.combine_first(
                pd.DataFrame.from_dict(data).T
            )
            self.data.to_csv('data/company_data/wlw_hildesheim.csv')
            self.next_page()

        return

    def random_sleep(self):
        # sleep random
        sleep_time = random.uniform(1, 3)
        time.sleep(sleep_time)

    def extract_company_info(self, soup):
        qinfo = self.extract_quick_info_box(soup=soup)
        portfolio = self.extract_portfolio(soup=soup)
        categories = self.extract_product_categories()

        portfolio.update(categories)
        qinfo.update(portfolio)
        return qinfo

    def extract_product_categories(self):
        category_page = 1

        categories = set()
        while True:
            headers = {
                'Accept-Language': 'de'
            }
            response = requests.get(
                self._create_categories_query_url(category_page),
                headers=headers
            ).json()
            categories = \
                categories | {iresponse['translated_name'] for iresponse in
                              response['company_categories']}

            if response['paging']['total_pages'] > category_page:
                category_page += 1
            else:
                break

        return {'product_categories': categories}

    def _create_categories_query_url(self, category_page):
        categories_query_url = \
            'https://api.visable.io/unified_search/v1/companies/%s' \
            '/categories?page=%i&per_page=30' % \
            (self._company_website.split('/')[-1], category_page)
        return categories_query_url

    def get_soup(self):
        page_content = requests.get(self.search_url)
        soup = BeautifulSoup(page_content.text,
                             features='lxml'
                             )
        return soup

    def get_company_websites(self, soup):
        websites = soup.find_all('a', {'data-test': 'company-name'})

        websites = [self.root_website + iwebsite.get('href')
                    for iwebsite in websites]
        return websites

    def extract_quick_info_box(self, soup):
        qinfo_box = soup.find('div', {
            'class': 'flex flex-col gap-2 lg:min-w-[250px] lg:max-w-[250px] '
                     'xl:min-w-[325px] xl:max-w-[325px]'
        }
                              )
        data = {}
        print('extracting page: %s' % self._company_website)

        data.update(self._get_company_name(qinfo_box))

        data.update(self._get_company_address(qinfo_box))

        data.update(self._get_general_info(qinfo_box))

        data.update(self._get_supplier_types(qinfo_box))

        data.update(self._get_website(qinfo_box))

        # ToDo: Phone number could not be found, since you have to click on
        #  a button, to make the number appear as an element. Maybe add the
        #  click, find the phonenumber some where else or dont add at all.

        return data

    @staticmethod
    def _get_website(qinfo_box):
        try:
            website = {
                'comapny_website':
                    qinfo_box.find('a', class_='company-name')['href']
            }
        except:  # noqa: E722
            print(ResourceWarning('No website found'))
            return {}

        return website

    @staticmethod
    def _get_supplier_types(qinfo_box):
        supplier_type_box = qinfo_box.find('div',
                                           {'data-test': 'supplier-types'}
                                           )
        try:
            supplier_types = \
                supplier_type_box.find_all('div', recursive=False)[1]
        except AttributeError:
            ResourceWarning('No supplier type box found')
            return {}
        supplier_types = supplier_types.find_all('div')
        data = {}
        for isupllier_type in supplier_types:
            data[isupllier_type.text] = True

        return data

    @staticmethod
    def _get_general_info(soup):
        """
        Extracts and returns general information from the specified HTML
        element.

        Parameters
        ----------
        soup : bs4.BeautifulSoup
            The BeautifulSoup object representing the HTML content of the
            quick info box.

        Returns
        -------
        dict
            A dictionary containing the extracted general information.

        Notes
        -----
        This function assumes that the general information is stored within
        a <div> element
        with the class 'flex flex-col md:flex-row lg:flex-col flex-wrap
        gap-2', containing multiple
        child <div> elements with the class 'flex gap-1 font-copy-400
        text-navy-70'.
        It iterates through these child elements, extracts information using
        the 'data-test' attribute
        as the key and the text within the <strong> tag as the value,
        and returns the information as a dictionary.

        Example Output
        -------
        {'info_key1': 'Info Value 1', 'info_key2': 'Info Value 2', ...}
        """

        isoup = soup.find('div', {'data-test': 'company-facts'})
        if isoup is None:
            ResourceWarning('no general information found in ')
            return {}
        isoup = isoup.findChildren('div', recursive=False)
        general_info = {}
        for iinfo in isoup:
            extracted_iinfo = {iinfo['data-test']: iinfo.find('strong').text}
            general_info.update(extracted_iinfo)

        return general_info

    @staticmethod
    def _get_company_address(qinfo_box):
        """
        Extracts and parses the company address from the specified HTML
        element.

        Parameters
        ----------
        qinfo_box : bs4.element.Tag
            The BeautifulSoup Tag representing the container of the
            company address information.

        Returns
        -------
        dict
            A dictionary containing the parsed components of the company
            address.

        Raises
        ------
        ValueError
            If the provided HTML structure does not match the expected
            format.

        Notes
        -----
        This function assumes that the company address is stored within
        a <div> element
        with the class 'font-copy-400' inside the given 'qinfo_box'. It
        extracts the address,
        separates it into street, ZIP code, and city components,
        and returns them as a dictionary.

        If the name is not found, {'company_street': None, 'company_zip': None,
        'company_city': None} will be returned

        Example Output
        -------
        {'company_street': 'Street Address', 'company_zip': 'ZIP Code',
        'company_city': 'City Name'}
        """
        address_soup = qinfo_box.find('div',
                                      {'class': 'p-2 flex flex-col h-full'}
                                      )
        address_soup = address_soup.findChildren('div', recursive=False)
        address_soup = address_soup[1].findChildren('div', recursive=False)
        company_address = address_soup[0].text

        if company_address is None:
            return {
                'company_street': None,
                'company_zip': None,
                'company_city': None
            }

        company_city, company_zip, company_street = \
            WlwCrawler.unpack_address(company_address)

        return {
            'company_street': company_street,
            'company_zip': company_zip,
            'company_city': company_city
        }

    @staticmethod
    def unpack_address(company_address):
        try:
            street, zip_city = company_address.split(',')
        except ValueError as exception_info:
            print(exception_info.args[0] +
                  '\nreturning dummy_address'
                  )
            return 'dummy', 'dummy', 'dummy'
        zip_city = zip_city.strip()

        *city, zipcode = zip_city.split(' ')

        city = ''.join(icity_part + ' ' for icity_part in city)
        city = city.strip()

        return city, zipcode, street

    @staticmethod
    def _get_company_name(qinfo_box):
        """
        Extracts and returns the company name from the specified HTML element.

        Parameters
        ----------
        qinfo_box : bs4.element.Tag
            The BeautifulSoup Tag representing the quick info box containing
            the
            company name.

        Returns
        -------
        dict
            A dictionary containing the extracted company name.

        Raises
        ------
        AttributeError
            If the provided HTML structure does not match the expected format.

        Notes
        -----
        This function assumes that the company name is stored within an <a>
        element with the class
        'company-name mt-2 text-navy-100 hover:no-underline' inside the given
        'qinfo_box'.
        It extracts the company name from the <h1> tag within the <a> element
        and returns it as a dictionary.

        If the name is not found, {'company_name': None} will be returned

        Example Output
        -------
        {'company_name': 'Company Name'}
        """
        company_name = qinfo_box.find('h1')
        if company_name is None:
            raise ValueError('No company name found. Search query: %s'
                             % 'h1'
                             )
        return {'company_name': company_name.text}

    def extract_portfolio(self, soup):
        product_websites = self._get_product_websites(soup)

        products = self._get_product_descriptions(product_websites)

        return {'portfolio': products}

    def _get_product_websites(self, soup, _page=1):
        portfolio_soup = soup.find('div', {'class': 'portfolio'})

        portfolio = portfolio_soup.find_all('div',
                                            {
                                                'class': 'product '
                                                         'rounded '
                                                         'bg-white '
                                                         'shadow-100 p-1'
                                            }
                                            )
        # The check here is to see if the company did not upload a portfolio
        # or if the portfolio items could not be found
        if not portfolio:
            if portfolio_soup.find('div', {'class': 'mb-2'}).text == \
                    'Der Anbieter hat noch keine Produkte hochgeladen.':
                pass
            else:
                raise ResourceWarning('Portfolio items could not be found')

        product_websites = [self.root_website +
                            iportfolio.find('a').get('href')
                            for iportfolio
                            in portfolio]

        if portfolio_soup.find(name='a', attrs={'class': 'button next'}):
            next_page = _page + 1

            next_soup = requests.get(self._company_website +
                                     '?page=%i' % next_page
                                     )
            next_soup = BeautifulSoup(next_soup.text,
                                      features='lxml'
                                      )

            product_websites += self._get_product_websites(soup=next_soup,
                                                           _page=next_page
                                                           )

        return product_websites

    @staticmethod
    def _get_product_descriptions(product_websites):
        products = {}
        for iwebsite in product_websites:
            content = requests.get(iwebsite)
            isoup = BeautifulSoup(content.text,
                                  features="lxml"
                                  )

            product_name = \
                isoup.find(
                    name='div',
                    attrs={'class': 'p-2 flex flex-col h-full'}
                ).find('h1').text.strip()

            product_description = \
                isoup.find(
                    name='div',
                    attrs={'class': 'p-2 md:p-3 flex flex-col h-full'}
                ).find_all('div')[-1].text

            products[product_name] = product_description

        return products

    def get_persisted_data(self, data_path):
        if os.path.exists(data_path):
            data = pd.read_csv(data_path, index_col=0)
        else:
            data = pd.DataFrame()

        return data


class WlwNameCrawler(WlwCrawler):
    def __init__(self, company_name, company_address=None):
        self.root_website = 'https://www.wlw.de'
        self.origin_name = company_name
        self.input_company_address = company_address

    def crawl_wlw_data(self):
        self.search_url = self.set_search_url()
        search_results_soup = self.get_search_results_soup()
        if len(search_results_soup.text) == 0:
            print(f'No search results for {self.origin_name}')
            return {"company_name": self.origin_name}

        most_similar_company_soup, similarity_score = \
            self.get_most_similar_company_soup(search_results_soup)

        company_website_soup = self.get_company_website_soup(
            most_similar_company_soup
        )
        data = self.extract_company_info(company_website_soup)
        data["company_city"], data["company_zip"], data["company_street"] = \
            self.unpack_address(self.input_company_address)

        if similarity_score > 7:
            print(f"Levenshtein distance {similarity_score}>5,"
                  f"returning None\n"
                  f"origin_name: {self.origin_name}\n"
                  f"most similar: {data["company_name"]}\n"
                  )
            return {"company_name": data["company_name"]}

        return data

    def get_company_website_soup(self, soup):
        self._company_website = self.root_website + soup.get('href')
        response = requests.get(self._company_website)
        return BeautifulSoup(response.text,
                             features='lxml'
                             )

    def set_search_url(self):
        encoded_query = urllib.parse.quote(self.origin_name)
        query = self.root_website + "/de/suche?isPserpFirst=1&q=" + \
                encoded_query
        return query

    def get_search_results_soup(self):
        response = requests.get(self.search_url)
        soup = BeautifulSoup(response.text,
                             features='lxml'
                             )
        soup = soup.find("div", {"data-test": "search-results"})
        return soup

    def get_most_similar_company_soup(self, search_results_soup):
        company_list_soup, company_names = \
            self._extract_company_list(search_results_soup)

        most_similar_index, similarity_score = \
            self._get_most_similar_names_position(company_names)

        return company_list_soup[most_similar_index], similarity_score

    @staticmethod
    def _extract_company_list(soup):
        company_names_soup = soup.find_all('a', {'data-test': 'company-name'})
        company_names = [iname.text for iname in company_names_soup]
        return company_names_soup, company_names

    def _get_most_similar_names_position(self, company_names):
        similarity_scores = self._calc_similarity(company_names)
        most_similar_index = similarity_scores.index(np.min(similarity_scores))

        return most_similar_index, np.min(similarity_scores)

    def _calc_similarity(self, company_names):
        similarity_scores = [textdistance.levenshtein(
            self.origin_name.lower(), icompany.lower()
        ) for icompany in company_names]
        return similarity_scores

    @staticmethod
    def extract_portfolio(soup):
        """
        portfolio is not used hence we will overwrite the other method to
        save some time

        Returns
        -------

        """
        return {}
