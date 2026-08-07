import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class DataPreprocessing:

    def __init__(self, city="Karachi"):

        self.city = city

        self.data = None

        self.scaler_x = MinMaxScaler(
            feature_range=(0, 1)
        )

        self.scaler_y = MinMaxScaler(
            feature_range=(0, 1)
        )

        self.x_train = None
        self.y_train = None

        self.x_val = None
        self.y_val = None

        self.x_test = None
        self.y_test = None

        self.feature_columns = [

            "tmin",
            "tmax",
            "humidity",
            "pressure",
            "wspd",
            "prcp",
            "dew_point",
            "cloud_cover",
            "visibility",

            "month_sin",
            "month_cos",

            "tavg_lag1",
            "tavg_lag7",
            "tavg_roll7"

        ]

        self.target_column = "tavg"


    # ==========================================================
    # LOAD DATA
    # ==========================================================

    def load_data(
        self,
        csv_path="../../datasets/pakistan_weather_2000_2024.csv"
    ):

        self.data = pd.read_csv(csv_path)

        print("\n" + "=" * 60)
        print("DATASET LOADED SUCCESSFULLY")
        print("=" * 60)

        print(
            "Dataset Shape:",
            self.data.shape
        )


    # ==========================================================
    # FILTER CITY
    # ==========================================================

    def filter_city(self):

        if "city" not in self.data.columns:

            raise ValueError(
                "Column 'city' not found in dataset."
            )

        self.data = self.data[
            self.data["city"].astype(str).str.strip().str.lower()
            == self.city.lower()
        ].copy()

        print(
            f"{self.city} Records:",
            len(self.data)
        )

        if len(self.data) == 0:

            raise ValueError(
                f"No records found for city: {self.city}"
            )


    # ==========================================================
    # CLEAN BASIC DATA
    # ==========================================================

    def clean_data(self):

        # Convert date

        self.data["date"] = pd.to_datetime(
            self.data["date"],
            errors="coerce"
        )


        # Remove rows where date is invalid

        self.data = self.data[
            self.data["date"].notna()
        ].copy()


        # Sort chronologically

        self.data = self.data.sort_values(
            by="date"
        ).reset_index(
            drop=True
        )


        print(
            "Rows after date cleaning:",
            len(self.data)
        )


        # Convert required weather columns to numeric

        numeric_columns = [

            "tmin",
            "tmax",
            "tavg",
            "humidity",
            "pressure",
            "wspd",
            "prcp",
            "dew_point",
            "cloud_cover",
            "visibility"

        ]


        for column in numeric_columns:

            if column in self.data.columns:

                self.data[column] = pd.to_numeric(
                    self.data[column],
                    errors="coerce"
                )


        # Show missing values before handling

        print("\nMissing Values Before Handling:")

        missing = self.data[
            numeric_columns
        ].isnull().sum()

        print(missing)


    # ==========================================================
    # HANDLE MISSING VALUES
    # ==========================================================

    def handle_missing_values(self):

        print("\nHandling Missing Values...")


        numeric_columns = [

            "tmin",
            "tmax",
            "tavg",
            "humidity",
            "pressure",
            "wspd",
            "prcp",
            "dew_point",
            "cloud_cover",
            "visibility"

        ]


        available_columns = [

            column
            for column in numeric_columns
            if column in self.data.columns

        ]


        # ------------------------------------------------------
        # Interpolate weather values based on time
        # ------------------------------------------------------

        self.data = self.data.set_index(
            "date"
        )


        self.data[available_columns] = (

            self.data[available_columns]
            .interpolate(
                method="time",
                limit_direction="both"
            )

        )


        # ------------------------------------------------------
        # Fill any remaining gaps
        # ------------------------------------------------------

        self.data[available_columns] = (

            self.data[available_columns]
            .ffill()
            .bfill()

        )


        self.data = self.data.reset_index()


        # ------------------------------------------------------
        # Target must exist
        # ------------------------------------------------------

        self.data = self.data[
            self.data[self.target_column].notna()
        ].copy()


        self.data = self.data.reset_index(
            drop=True
        )


        print(
            "\nMissing Values After Handling:"
        )

        print(
            self.data[available_columns]
            .isnull()
            .sum()
        )


        print(
            "\nRows after missing-value handling:",
            len(self.data)
        )


        if len(self.data) == 0:

            raise ValueError(
                "No data remains after missing-value handling."
            )


    # ==========================================================
    # FEATURE ENGINEERING
    # ==========================================================

    def create_features(self):

        # ------------------------------------------------------
        # Month
        # ------------------------------------------------------

        month = self.data["date"].dt.month


        # ------------------------------------------------------
        # Cyclic seasonal features
        # ------------------------------------------------------

        self.data["month_sin"] = np.sin(
            2 * np.pi * month / 12
        )

        self.data["month_cos"] = np.cos(
            2 * np.pi * month / 12
        )


        # ------------------------------------------------------
        # Lag features
        # ------------------------------------------------------

        self.data["tavg_lag1"] = (

            self.data["tavg"]
            .shift(1)

        )


        self.data["tavg_lag7"] = (

            self.data["tavg"]
            .shift(7)

        )


        # ------------------------------------------------------
        # 7-day rolling average
        # ------------------------------------------------------

        self.data["tavg_roll7"] = (

            self.data["tavg"]
            .rolling(
                window=7,
                min_periods=7
            )
            .mean()

        )


        # ------------------------------------------------------
        # Remove only rows created as NaN by lag/rolling
        # ------------------------------------------------------

        self.data = self.data.dropna(
            subset=[
                "tavg_lag1",
                "tavg_lag7",
                "tavg_roll7"
            ]
        ).reset_index(
            drop=True
        )


        print(
            "\nFeature Engineering Completed"
        )

        print(
            "Rows after features:",
            len(self.data)
        )


        if len(self.data) == 0:

            raise ValueError(
                "No data remains after feature engineering."
            )


    # ==========================================================
    # HANDLE OUTLIERS
    # ==========================================================

    def remove_outliers(self):

        print(
            "\nBefore Outlier Handling:",
            self.data.shape
        )


        columns = [

            "humidity",
            "pressure",
            "wspd",
            "visibility",
            "prcp"

        ]


        for column in columns:

            if column not in self.data.columns:
                continue


            # Use 1st and 99th percentile

            lower = self.data[
                column
            ].quantile(0.01)


            upper = self.data[
                column
            ].quantile(0.99)


            # Clip instead of deleting rows

            self.data[column] = np.clip(
                self.data[column],
                lower,
                upper
            )


        print(
            "Outliers handled using percentile clipping"
        )


        print(
            "After Outlier Handling:",
            self.data.shape
        )


    # ==========================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # ==========================================================

    def split_data(self):

        total_rows = len(self.data)


        if total_rows < 100:

            raise ValueError(
                "Dataset is too small for LSTM training."
            )


        # 70% training

        train_size = int(
            total_rows * 0.70
        )


        # 15% validation

        val_size = int(
            total_rows * 0.15
        )


        # Chronological split

        train = self.data.iloc[
            :train_size
        ].copy()


        val = self.data.iloc[
            train_size:
            train_size + val_size
        ].copy()


        test = self.data.iloc[
            train_size + val_size:
        ].copy()


        print("\n" + "=" * 60)
        print("DATA SPLIT")
        print("=" * 60)


        print(
            "Train:",
            train.shape
        )


        print(
            "Validation:",
            val.shape
        )


        print(
            "Test:",
            test.shape
        )


        return train, val, test


    # ==========================================================
    # SCALE DATA
    # ==========================================================

    def scale_data(
        self,
        train,
        val,
        test
    ):

        # ------------------------------------------------------
        # Features
        # ------------------------------------------------------

        X_train = train[
            self.feature_columns
        ].copy()


        X_val = val[
            self.feature_columns
        ].copy()


        X_test = test[
            self.feature_columns
        ].copy()


        # ------------------------------------------------------
        # Target
        # ------------------------------------------------------

        y_train = train[
            [self.target_column]
        ].copy()


        y_val = val[
            [self.target_column]
        ].copy()


        y_test = test[
            [self.target_column]
        ].copy()


        # ------------------------------------------------------
        # Safety checks
        # ------------------------------------------------------

        if len(X_train) == 0:

            raise ValueError(
                "Training dataset is empty."
            )


        if len(X_val) == 0:

            raise ValueError(
                "Validation dataset is empty."
            )


        if len(X_test) == 0:

            raise ValueError(
                "Testing dataset is empty."
            )


        # ------------------------------------------------------
        # Scale X
        #
        # IMPORTANT:
        # Fit scaler ONLY on training data
        # ------------------------------------------------------

        X_train = self.scaler_x.fit_transform(
            X_train
        )


        X_val = self.scaler_x.transform(
            X_val
        )


        X_test = self.scaler_x.transform(
            X_test
        )


        # ------------------------------------------------------
        # Scale Y
        # ------------------------------------------------------

        y_train = self.scaler_y.fit_transform(
            y_train
        )


        y_val = self.scaler_y.transform(
            y_val
        )


        y_test = self.scaler_y.transform(
            y_test
        )


        print("\nScaling Completed")


        print(
            "X Train:",
            X_train.shape
        )


        print(
            "Y Train:",
            y_train.shape
        )


        return (

            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test

        )


    # ==========================================================
    # CREATE LSTM SEQUENCES
    # ==========================================================

    def create_sequences(
        self,
        X,
        y,
        time_step=30
    ):

        X_sequences = []
        y_sequences = []


        if len(X) <= time_step:

            raise ValueError(
                "Not enough data to create LSTM sequences."
            )


        for i in range(
            len(X) - time_step
        ):

            X_sequences.append(
                X[
                    i:
                    i + time_step
                ]
            )


            y_sequences.append(
                y[
                    i + time_step
                ]
            )


        return (

            np.array(
                X_sequences,
                dtype=np.float32
            ),

            np.array(
                y_sequences,
                dtype=np.float32
            )

        )


    # ==========================================================
    # COMPLETE PIPELINE
    # ==========================================================

    def run(self):

        # ------------------------------------------------------
        # 1. Load
        # ------------------------------------------------------

        self.load_data()


        # ------------------------------------------------------
        # 2. Filter Karachi
        # ------------------------------------------------------

        self.filter_city()


        # ------------------------------------------------------
        # 3. Basic cleaning
        # ------------------------------------------------------

        self.clean_data()


        # ------------------------------------------------------
        # 4. Missing values
        # ------------------------------------------------------

        self.handle_missing_values()


        # ------------------------------------------------------
        # 5. Feature engineering
        # ------------------------------------------------------

        self.create_features()


        # ------------------------------------------------------
        # 6. Handle outliers
        # ------------------------------------------------------

        self.remove_outliers()


        # ------------------------------------------------------
        # 7. Train / validation / test
        # ------------------------------------------------------

        train, val, test = self.split_data()


        # ------------------------------------------------------
        # 8. Scaling
        # ------------------------------------------------------

        (
            X_train,
            y_train,

            X_val,
            y_val,

            X_test,
            y_test

        ) = self.scale_data(
            train,
            val,
            test
        )


        # ------------------------------------------------------
        # 9. Create sequences
        # ------------------------------------------------------

        self.x_train, self.y_train = (
            self.create_sequences(
                X_train,
                y_train,
                time_step=30
            )
        )


        self.x_val, self.y_val = (
            self.create_sequences(
                X_val,
                y_val,
                time_step=30
            )
        )


        self.x_test, self.y_test = (
            self.create_sequences(
                X_test,
                y_test,
                time_step=30
            )
        )


        # ------------------------------------------------------
        # Final information
        # ------------------------------------------------------

        print("\n" + "=" * 60)
        print("DATA PREPROCESSING COMPLETED")
        print("=" * 60)


        print(
            "\nTraining X Shape:",
            self.x_train.shape
        )


        print(
            "Training Y Shape:",
            self.y_train.shape
        )


        print(
            "\nValidation X Shape:",
            self.x_val.shape
        )


        print(
            "Validation Y Shape:",
            self.y_val.shape
        )


        print(
            "\nTesting X Shape:",
            self.x_test.shape
        )


        print(
            "Testing Y Shape:",
            self.y_test.shape
        )


        print(
            "\nNumber of Features:",
            len(self.feature_columns)
        )


        print(
            "Time Step:",
            30
        )


        # ------------------------------------------------------
        # Return everything required by training
        # ------------------------------------------------------

        return (

            self.x_train,
            self.y_train,

            self.x_val,
            self.y_val,

            self.x_test,
            self.y_test,

            self.scaler_x,
            self.scaler_y

        )