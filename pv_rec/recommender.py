import pandas as pd
import time

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from scipy.cluster import hierarchy
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

from structlog import get_logger


log=get_logger()


class Recommender:
    def __init__(self):
        self.cut_line=1.5

        self.model=None
        self.scaler=MinMaxScaler()

        self.ml_data=None
        self.column_shema=None
        self.label_averages=None
        self.pv_affinity_scores=None

    def fit(self, wlw_data: pd.DataFrame, mastr_data: pd.DataFrame,
            mapping_needed: bool=True,
            method='ward', metric='euclidean'):
        """
        Fit the recommender model using the provided company data.

        :param products:
        :return:
        """
        start_time=time.time()
        log.info("Preparing data")
        if 'product_categories' in wlw_data.columns or \
           'product_categories' in mastr_data.columns:
            log.info("Dropping product categories")
            data_a=mastr_data.drop(['product_categories'], axis=1).copy()
            data_b=wlw_data.drop(['product_categories'], axis=1).copy()
        else:
            data_a=mastr_data.copy()
            data_b=wlw_data.copy()

        self.ml_data=data_a.combine_first(data_b).copy()
        self.ml_data.sort_index(inplace=True, axis=1)

        # for later identification
        mastr_data["source"]="mastr"
        wlw_data["source"]="wlw"

        if mapping_needed:
            log.info("Map categorical data")
            self.ml_data=self._map_ordinal_data(self.ml_data)

        log.info("handle NaNs")
        self.ml_data.fillna(0, inplace=True)

        log.info("Scaling data")
        self.scaler.fit(self.ml_data)
        scaled_ml=self.scaler.transform(self.ml_data)

        log.info("create linkage tree")
        self.model=linkage(scaled_ml,
                           method=method, metric=metric,
                           optimal_ordering=True
                           )

        log.info("Get cluster labels")
        cluster_labels=fcluster(self.model,
                                t=self.cut_line,
                                criterion='distance'
                                )
        self.ml_data["labels"]=cluster_labels

        log.info("Get number of Mastr companies in clusters")
        results={}
        for icluster_num in self.ml_data.labels:
            companies_in_cluster=self.ml_data[
                self.ml_data["labels"]==icluster_num].copy()
            num_mastr_companises=companies_in_cluster.index.intersection(
                mastr_data.index
                )

            results[f"cluster_{icluster_num}"]={
                'total_companies': len(companies_in_cluster.index),
                'mastr_data': len(num_mastr_companises),
                'percent': len(num_mastr_companises) / len(
                    companies_in_cluster.index
                    )
                }

        log.info("Get PV affinity scores")
        self.pv_affinity_scores=pd.DataFrame(results).T

        end_time=time.time()
        log.info("Model fitted", duration=end_time - start_time)

        log.info("Store column schema")
        self.column_shema=self.ml_data.columns

        log.info("get all feature averages for clusters")
        self.label_averages=pd.concat([
            pd.DataFrame(
                scaled_ml,
                columns=self.ml_data.columns[:-1],
                index=self.ml_data.index
                ),
            self.ml_data.labels], axis=1
            ).groupby('labels').mean()

        return self.pv_affinity_scores

    def recommend(self, company_data: pd.DataFrame, mapping_needed: bool=True):
        new_company=company_data.copy()

        if mapping_needed:
            log.info("map categorical data")
            new_company=self._map_ordinal_data(new_company)

        log.info("handle nans")
        new_company.fillna(0, inplace=True)

        log.info("set columns in correct order")
        if "product_categories" in new_company.columns:
            new_company.drop("product_categories", axis=1, inplace=True)

        for column in self.column_shema:
            if column not in new_company.columns and \
                column!="labels":
                new_company[column]=0
        new_company=new_company.reindex(sorted(new_company.columns), axis=1)

        log.info("scale data")
        new_company_scaled=self.scaler.transform(new_company)
        # new_company_scaled = new_company_scaled[]

        log.info("Get closest cluster to new data")
        closest_cluster=[]
        for inew_company in new_company_scaled:
            distance=[euclidean(inew_company, label_avg) for label_avg in
                      self.label_averages.values]
            closest_cluster.append(np.argmin(distance))

        new_company['labels']=closest_cluster
        # 0 means no cluster found
        # ToDo: find out why this happens, there should always be a cluster
        new_company=new_company[new_company.labels!=0]

        pv_affinity=[self.calc_pv_affinity_score(icluster)
                     for icluster in new_company.labels]
        pv_affinity=pd.DataFrame(pv_affinity, index=new_company.index,
                                 columns=['pv_affinity']
                                 )

        return pv_affinity, new_company

    def calc_pv_affinity_score(self, cluster_number):
        return self.pv_affinity_scores \
                   .loc[f'cluster_{cluster_number}', :].percent

    def recall(self, affinity):
        threshold=0.5

        p=len(affinity)
        tp=np.sum(affinity.pv_affinity >= threshold)
        fn=np.sum(affinity.pv_affinity < threshold)

        # ToDo: Maybe add missrate as well
        # Trade-off between recall and missrate would be interesting
        recall=tp / p
        return recall

    def _map_ordinal_data(self, data: pd.DataFrame):
        distribution_area_map={
            "unknown": 0,
            "Lokal": 1,
            "Regional": 2,
            "National": 3,
            "Europa": 4,
            "Weltweit": 5
            }
        employee_count_map={
            "unknown": 0,
            "1-4": 1,
            "5-9": 2,
            "10-19": 3,
            "20-49": 4,
            "50-99": 5,
            "100-199": 6,
            "200-499": 7,
            "500-999": 8,
            "1000+": 9,
            }

        data.loc[:, 'distribution_area']=data['distribution_area'].map(
            distribution_area_map
            )
        data.loc[:, 'employee_count']=data['employee_count'].map(
            employee_count_map
            )

        return data

    # %% plots
    def plot(self):
        cmap=plt.cm.Spectral(np.linspace(0, 1, 15))
        hierarchy.set_link_color_palette(
            [mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap]
            )

        fig, ax=plt.subplots(figsize=(12, 7), ncols=1, nrows=2,
                             gridspec_kw={'height_ratios': [0.6, 0.4]}
                             )
        dendrogram(self.model,
                   truncate_mode='level',
                   no_labels=True,
                   color_threshold=self.cut_line,
                   ax=ax[0]
                   )
        ax[0].hlines(self.cut_line, -1, 30000, 'grey', linewidth=1.5,
                     linestyles='--'
                     )
        dendrogram(self.model,
                   truncate_mode='level',
                   no_labels=True,
                   color_threshold=self.cut_line,
                   ax=ax[1]
                   )
        ax[1].hlines(self.cut_line, -1, 30000, 'grey', linewidth=1.5,
                     linestyles='--'
                     )

        ax[1].set_ylim(0, 2)

        fig.text(0.5, 0.05, 'Companies', ha='center', va='center')
        fig.text(0.05, 0.5, 'Euclidean distance [-]', ha='center', va='center',
                 rotation='vertical'
                 )

        plt.tight_layout(rect=(0.05, 0.05, 1, 1))

    def heatmap(self, cluster_id: int, pv_affinity_score: pd.DataFrame):
        company_data=self.ml_data.copy()
        grouped_companies=company_data[
            company_data["labels"]==cluster_id].copy()
        grouped_companies.replace({False: 0, True: 1}, inplace=True)

        data_to_plot=grouped_companies
        scaler=MinMaxScaler()

        data_to_plot=scaler.fit_transform(data_to_plot)

        plt.figure(figsize=(16, 16))
        plt.imshow(
            data_to_plot
            , cmap='viridis_r'
            )

        plt.xlabel('features')
        plt.ylabel('companies')

        plt.colorbar(orientation='horizontal')
        plt.show()

        plt.hist(pv_affinity_score.iloc[:, -1])

    def plot_recall(self, pv_affinity, recall, ax=None):
        if ax is None:
            fig, ax=plt.subplots(figsize=(8, 5))
        ax.hist(pv_affinity,bins=np.arange(0, 1.2, 0.1) - 0.05, rwidth=1)

        ax.set_title('Recall: %.2f' % (recall*100), fontsize=12)
        ax.set_xlim((-0.05, 1.15))
