import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class DataReader:
    """
    This class is responsible for loading, preprocessing, and preparing data for training and testing neural network models.

    It handles data loading from specified paths, feature selection, normalization, and conversion to PyTorch tensors, facilitating the data preparation process for machine learning models.

    """

    # A predefined list of all possible feature names.
    # Please DO NOT modify this list. Only complete or modify the _find_important_features method. Do not alter any other class methods or features in this file.
    ALL_FEATURE_LIST = ['Pd_bus1', 'Qd_bus1', 'Pd_bus2', 'Qd_bus2', 'Pd_bus3', 'Qd_bus3', 'Pd_bus4',
                        'Qd_bus4', 'Pd_bus5', 'Qd_bus5', 'Pd_bus6', 'Qd_bus6', 'Pd_bus7', 'Qd_bus7',
                        'Pd_bus8', 'Qd_bus8', 'Pd_bus9', 'Qd_bus9', 'Pd_bus10', 'Qd_bus10', 'Pd_bus11',
                        'Qd_bus11', 'Pd_bus12', 'Qd_bus12', 'Pd_bus13', 'Qd_bus13', 'Pd_bus14', 'Qd_bus14',
                        'Pd_bus15', 'Qd_bus15', 'Pd_bus16', 'Qd_bus16', 'Pd_bus17', 'Qd_bus17', 'Pd_bus18',
                        'Qd_bus18', 'Pd_bus19', 'Qd_bus19', 'Pd_bus20', 'Qd_bus20', 'Pd_bus21', 'Qd_bus21',
                        'Pd_bus22', 'Qd_bus22', 'Pd_bus23', 'Qd_bus23', 'Pd_bus24', 'Qd_bus24', 'Pd_bus25',
                        'Qd_bus25', 'Pd_bus26', 'Qd_bus26', 'Pd_bus27', 'Qd_bus27', 'Pd_bus28', 'Qd_bus28',
                        'Pd_bus29', 'Qd_bus29', 'Pd_bus30', 'Qd_bus30', 'Pd_bus31', 'Qd_bus31', 'Pd_bus32',
                        'Qd_bus32', 'Pd_bus33', 'Qd_bus33', 'Pd_bus34', 'Qd_bus34', 'Pd_bus35', 'Qd_bus35',
                        'Pd_bus36', 'Qd_bus36', 'Pd_bus37', 'Qd_bus37', 'Pd_bus38', 'Qd_bus38', 'Pd_bus39', 'Qd_bus39',
                        'Pg_gen1', 'Qg_gen1', 'Pg_gen2', 'Qg_gen2', 'Pg_gen3', 'Qg_gen3', 'Pg_gen4', 'Qg_gen4',
                        'Pg_gen5', 'Qg_gen5', 'Pg_gen6', 'Qg_gen6', 'Pg_gen7', 'Qg_gen7', 'Pg_gen8', 'Qg_gen8',
                        'Pg_gen9', 'Qg_gen9', 'Pg_gen10', 'Qg_gen10']

    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Initializes the DataReader with paths to the feature and label datasets.

        Parameters:
            input_path (str): The file path to the features dataset.
            output_path (str): The file path to the labels dataset.
        """
        self.features_path: str = input_path  # The file path to the features dataset.
        self.labels_path: str = output_path  # The file path to the labels dataset.
        self.features_df: pd.DataFrame = None  # The loaded features data as a pandas DataFrame.
        self.labels_df: pd.DataFrame = None  # The loaded labels data as a pandas DataFrame.
        self.scaler: StandardScaler = StandardScaler()  # An instance of sklearn's StandardScaler for data normalization.
        self.X_normalized: np.ndarray = None  # The normalized feature data.
        self.X_tensor: torch.FloatTensor = None  # The feature data converted to a PyTorch FloatTensor.
        self.y_tensor: torch.FloatTensor = None  # The label data converted to a PyTorch FloatTensor.
        self.feature_dim: int = None  # The dimension of the feature space after preprocessing.
        self.label_dim: int = None  # The dimension of the label space.
        self.selected_feature_columns: list = None  # The columns selected as relevant (most important) features.

    def load_train_data(self) -> None:
        """
        Loads the training data from files, selects important features, normalizes the data, and converts it to tensors.
        
        This method orchestrates the data loading and preprocessing steps for training data, including feature selection, normalization, and tensor conversion.
        """
        self._load_data()
        self.selected_feature_columns = self._find_important_features(df_feature=self.features_df,
                                                                      df_labels=self.labels_df)
        logging.debug(f"{len(self.selected_feature_columns)} out of {len(DataReader.ALL_FEATURE_LIST)} selected.")
        logging.debug("Selected columns are: \n" + str(self.selected_feature_columns))
        # Assert that all elements in selected_columns are in ALL_FEATURE_LIST
        assert all(item in DataReader.ALL_FEATURE_LIST for item in
                   self.selected_feature_columns), "Not all elements of selected_columns are in ALL_FEATURE_LIST"
        self._reduce_feature_space()
        self._set_feature_and_label_dim()
        self._find_scaler_and_normalize_data()
        self._convert_to_tensor()

    def load_test_data(self, train_scaler: StandardScaler, selected_feature_columns: list) -> None:
        """
        Loads the test data from files, applies the training data's scaler and selected features, and converts it to tensors.

        Parameters:
            train_scaler (StandardScaler): The scaler used to normalize the training data.
            selected_feature_columns (list of str): The feature columns selected during training.
        """
        self._load_data()
        self.selected_feature_columns = selected_feature_columns
        self._reduce_feature_space()
        self._set_feature_and_label_dim()
        self._normalize_data(train_scaler)
        self._convert_to_tensor()

    def _load_data(self) -> None:
        """
        Private method to load feature and label data from the specified CSV files into pandas DataFrames.
        """
        self.features_df = pd.read_csv(self.features_path)
        self.labels_df = pd.read_csv(self.labels_path)

    def _find_scaler_and_normalize_data(self) -> None:
        """
        Private method to fit a scaler to the feature data and normalize it.
        """
        self.X_normalized = self.scaler.fit_transform(self.features_df)

    def _normalize_data(self, train_scaler: StandardScaler) -> None:
        """
        Private method to normalize the feature data using the provided scaler.

        Parameters:
            train_scaler (StandardScaler): The scaler used to normalize the training data.
        """
        self.X_normalized = train_scaler.transform(self.features_df)

    def _convert_to_tensor(self) -> None:
        """
        Private method to convert the normalized feature data and labels into PyTorch tensors.
        """
        self.X_tensor = torch.FloatTensor(self.X_normalized)
        self.y_tensor = torch.FloatTensor(self.labels_df.values)

    def _find_important_features(self, df_feature: pd.DataFrame, df_labels: pd.DataFrame) -> list:
        """
        Private method to identify and select important features from the dataset.

        Parameters:
            df_feature (DataFrame): The feature data as a pandas DataFrame.
            df_labels (DataFrame): The label data as a pandas DataFrame.
        Returns:
            selected_columns (list): The list of selected feature column names that are most important for the model
        """
        # Please only complete or modify this method, i.e., _find_important_features method. Do not alter any other class methods or features in this file.

        srs_constant_columns = df_feature.nunique() == 1  # Series of columns that have constant values. From pandas docs [1]. Last modified 03/23/2025.
        
        df_non_constant = df_feature.loc[:, ~srs_constant_columns] # Pull only non-constant columns from data. From pandas docs [2]. Last modified 03/23/2025.

        srs_std_devs = df_non_constant.std() # Series of standard deviations for non constant columns. From pandas docs [3]. Last modified 03/23/2025.

        num_extremes = 2 # Number of extremes to remove. Last modified 03/23/2025.
        std_devs_length = len(srs_std_devs) # Length of standard deviations series. From python docs [4]. Last modified 03/23/2025.
        
        if (std_devs_length >= ((num_extremes * 2) + 1)): # Check if there are enough columns to remove extremes. Last modified 03/23/2025.
            lowest_std_columns = srs_std_devs.nsmallest(min(num_extremes, std_devs_length)).index.to_list()
            highest_std_columns = srs_std_devs.nlargest(min(num_extremes, std_devs_length)).index.to_list()
        else:
            lowest_std_columns = []
            highest_std_columns = []

        # Combine all columns to drop
        columns_to_drop = highest_std_columns + lowest_std_columns

        # Drop highest & lowest std dev columns (after removing constant ones)
        df_train_features_filtered = df_non_constant.drop(columns=columns_to_drop)

        # Get selected columns
        selected_columns = df_train_features_filtered.columns.to_list()
        
        return selected_columns

    def _reduce_feature_space(self) -> None: # Description below. From assignment template repository [10]. Last modified 07/01/2024.
        """
        Private method to reduce the feature space to only the selected important features.
        """
        self.features_df = self.features_df[self.selected_feature_columns] # Select important columns from features_df. From assignment template repository [10]. Last modified 07/01/2024.

    def _set_feature_and_label_dim(self) -> None:
        """
        Private method to set the dimensions of the feature and label data after preprocessing.
        """
        self.feature_dim = self.features_df.shape[1]
        self.label_dim = self.labels_df.shape[1]
        
# References:
# [1] Pandas Developers, "pandas.DataFrame.nunique", 2024. Available: https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.DataFrame.nunique.html
# [2] Pandas Developers, "pandas.DataFrame.loc", 2024. Available: https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.DataFrame.loc.html
# [3] Pandas Developers, "pandas.DataFrame.std", 2024. Available: https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.DataFrame.std.html
# [4] Python Developers, "len()", 2024. Available: https://docs.python.org/3.12/library/functions.html#len


# [2] Pandas Developers, "pandas.DataFrame.columns", 2024. Available: https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.DataFrame.columns.html
# [3] Pandas Developers, "pandas.Index.to_list", 2024. Available: https://pandas.pydata.org/pandas-docs/version/2.2.2/reference/api/pandas.Index.to_list.html


# [10] Gholami, A., Shekari, T., "Intro to CPS Security - Mini Project 4", 2024. Available: https://github.com/tshekari3/cps_sec_mp4/tree/main