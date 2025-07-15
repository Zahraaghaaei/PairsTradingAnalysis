import sys
import functools
import operator
import collections
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

class DataProcessor:
    def __init__(self):
        pass

    def split_data(self, df_prices, training_dates, testing_dates, remove_nan=True):
        """
        This function splits a dataframe into training and validation sets
        :param df_prices: dataframe containing prices for all dates
        :param training_dates: tuple (training initial date, training final date)
        :param testing_dates: tuple (testing initial date, testing final date)
        :param remove_nan: flag to detect if nan values are to be removed

        :return: df with training prices
        :return: df with testing prices
        """
        if remove_nan:
            dataset_mask = ((df_prices.index >= training_dates[0]) &\
                            (df_prices.index <= testing_dates[1]))
            df_prices_dataset = df_prices[dataset_mask]
            print('Total of {} tickers'.format(df_prices_dataset.shape[1]))
            df_prices_dataset_without_nan = self.remove_tickers_with_nan(df_prices_dataset, 40)
            print('Total of {} tickers after removing tickers with Nan values'.format(
                df_prices_dataset_without_nan.shape[1]))
            df_prices = df_prices_dataset_without_nan.copy()

        train_mask = (df_prices.index <= training_dates[1])
        test_mask = (df_prices.index >= testing_dates[0])
        df_prices_train = df_prices[train_mask]
        df_prices_test = df_prices[test_mask]

        return df_prices_train, df_prices_test

    def remove_tickers_with_nan(self, df, threshold):
        """
        Removes columns with more than threshold null values
        """
        null_values = df.isnull().sum()
        null_values = null_values[null_values > 0]

        to_remove = list(null_values[null_values > threshold].index)
        df = df.drop(columns=to_remove)

        return df

    def get_candidate_pairs(self, clustered_series, pricing_df_train, pricing_df_test, 
                          p_value_threshold=0.05, subsample=0):
        """
        This function looks for tradable pairs over the clusters formed previously.

        :param clustered_series: series with cluster label info
        :param pricing_df_train: df with price series from train set
        :param pricing_df_test: df with price series from test set
        :param n_clusters: number of clusters
        :param p_value_threshold: p_value to check during cointegration test

        :return: list of pairs and its info
        :return: list of unique tickers identified in the candidate pairs universe
        """

        total_pairs, total_pairs_fail_criteria = [], []
        n_clusters = len(clustered_series.value_counts())
        for clust in range(n_clusters):
            sys.stdout.write("\r"+'Cluster {}/{}'.format(clust+1, n_clusters))
            sys.stdout.flush()
            symbols = list(clustered_series[clustered_series == clust].index)
            cluster_pricing_train = pricing_df_train[symbols]
            cluster_pricing_test = pricing_df_test[symbols]
            pairs, pairs_fail_criteria = self.find_pairs(cluster_pricing_train,
                                                      cluster_pricing_test,
                                                      p_value_threshold,
                                                      subsample)
            total_pairs.extend(pairs)
            total_pairs_fail_criteria.append(pairs_fail_criteria)

        print('Found {} pairs'.format(len(total_pairs)))
        unique_tickers = np.unique([(element[0], element[1]) for element in total_pairs])
        print('The pairs contain {} unique tickers'.format(len(unique_tickers)))

        # discarded
        review = dict(functools.reduce(operator.add, map(collections.Counter, total_pairs_fail_criteria)))
        print('Pairs Selection failed stage: ', review)

        return total_pairs, unique_tickers

    def find_pairs(self, data_train, data_test, p_value_threshold, subsample=0):
        """
        This function receives a df with the different securities as columns, and aims to find tradable
        pairs within this world. There is a df containing the training data and another one containing test data
        Tradable pairs are those that verify:
            - cointegration
            - minimium half life
            - minimium zero crossings

        :param data_train: df with training prices in columns
        :param data_test: df with testing prices in columns
        :param p_value_threshold:  pvalue threshold for a pair to be cointegrated
        :param min_half_life: minimium half life value of the spread to consider the pair
        :param min_zero_crossings: minimium number of allowed zero crossings
        :param hurst_threshold: mimimium acceptable number for hurst threshold
        :return: pairs that passed test
        """
        n = data_train.shape[1]
        keys = data_train.keys()
        pairs_fail_criteria = {'cointegration': 0, 'None': 0}
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                S1_train = data_train[keys[i]]; S2_train = data_train[keys[j]]
                S1_test = data_test[keys[i]]; S2_test = data_test[keys[j]]
                result, criteria_not_verified = self.check_properties((S1_train, S2_train), (S1_test, S2_test), p_value_threshold, subsample)
                pairs_fail_criteria[criteria_not_verified] += 1
                if result is not None:
                    pairs.append((keys[i], keys[j], result))

        return pairs, pairs_fail_criteria

    def check_properties(self, train_series, test_series, p_value_threshold, subsample=0):
        """
        Gets two time series as inputs and provides information concerning cointegration stasttics
        Y - b*X : Y is dependent, X is independent
        """

        # for some reason is not giving right results
        # t_statistic, p_value, crit_value = coint(X,Y, method='aeg')

        # perform test manually in both directions
        X = train_series[0]
        Y = train_series[1]
        pairs = [(X, Y), (Y, X)]
        pair_stats = [0] * 2
        criteria_not_verified = 'cointegration'

        # first of all, must verify price series S1 and S2 are I(1)
        stats_Y = self.check_for_stationarity(np.asarray(Y), subsample=subsample)
        if stats_Y['p_value'] > 0.10:
            stats_X = self.check_for_stationarity(np.asarray(X), subsample=subsample)
            if stats_X['p_value'] > 0.10:
                # conditions to test cointegration verified

                for i, pair in enumerate(pairs):
                    S1 = np.asarray(pair[0])
                    S2 = np.asarray(pair[1])
                    S1_c = sm.add_constant(S1)

                    # Y = bX + c
                    # ols: (Y, X)
                    results = sm.OLS(S2, S1_c).fit()
                    b = results.params[1]

                    if b > 0:
                        spread = pair[1] - b * pair[0] # as Pandas Series
                        spread_array = np.asarray(spread) # as array for faster computations

                        stats = self.check_for_stationarity(spread_array, subsample=subsample)
                        if stats['p_value'] < p_value_threshold:  # verifies required pvalue
                            criteria_not_verified = 'None'

                            pair_stats[i] = {'t_statistic': stats['t_statistic'],
                                              'critical_val': stats['critical_values'],
                                              'p_value': stats['p_value'],
                                              'coint_coef': b,
                                              'spread': spread,
                                              'Y_train': pair[1],
                                              'X_train': pair[0]
                                              }

        if pair_stats[0] == 0 and pair_stats[1] == 0:
            result = None
            return result, criteria_not_verified

        elif pair_stats[0] == 0:
            result = 1
        elif pair_stats[1] == 0:
            result = 0
        else: # both combinations are possible
            # select lowest t-statistic as representative test
            if abs(pair_stats[0]['t_statistic']) > abs(pair_stats[1]['t_statistic']):
                result = 0
            else:
                result = 1

        if result == 0:
            result = pair_stats[0]
            result['X_test'] = test_series[0]
            result['Y_test'] = test_series[1]
        elif result == 1:
            result = pair_stats[1]
            result['X_test'] = test_series[1]
            result['Y_test'] = test_series[0]

        return result, criteria_not_verified

    def check_for_stationarity(self, X, subsample=0):
        """
        H_0 in adfuller is unit root exists (non-stationary).
        We must observe significant p-value to convince ourselves that the series is stationary.

        :param X: time series
        :param subsample: boolean indicating whether to subsample series
        :return: adf results
        """
        if subsample != 0:
            frequency = round(len(X)/subsample)
            subsampled_X = X[0::frequency]
            result = adfuller(subsampled_X)
        else:
            result = adfuller(X)
        # result contains:
        # 0: t-statistic
        # 1: p-value
        # others: please see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

        return {'t_statistic': result[0], 'p_value': result[1], 'critical_values': result[4]}
