import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


class ElasticNetDeviationFromBaseFare(object):

    def __init__(self, boost_df: pd.DataFrame):
        self.RANDOM_STATE = 99
        self.df = boost_df
        self.df_agg = None
        self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None, None, None
        self.X_train_scaled, self.X_test_scaled = None, None
        self.mae, self.mse, self.r2 = None, None, None
        self.elastic_net = None


    def clean_data(self):
        # Parse date columns as datetime
        date_columns = ["boost_timestamp", "created_at", "claimed_at", "scheduled_starts_at", "scheduled_ends_at",
                        "unclaimed_at", "trip_completed_at"]
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], format="ISO8601")
        # Remove the five trip_ids that have null values in  commute_distance
        self.df = self.df[self.df.trip_id.isin(self.df[self.df.commute_distance.notnull()].trip_id.values)]
        # Drop trip ids with single_boost_amount_cents of 0 when the trip was boosted
        trips_boost_yes_amount_zero = self.df[
            (self.df.boost_ind == 1) & (self.df.single_boost_amount_cents == 0)].trip_id.values
        self.df = self.df[~self.df.trip_id.isin(trips_boost_yes_amount_zero)]
        # Drop trip ids where schedule start date is less than claimed date
        self.df = self.df[self.df.scheduled_starts_at > self.df.claimed_at]

    @staticmethod
    def assign_base_fare(row):
        return row["dollars_paid_to_driver"] if row["boost_ind"] == 0 else row["dollars_paid_to_driver"] - row[
            "total_boost_dollars"]

    def aggregate_for_analysis(self):
        # Take the max over columns of interest because same value is repeated for each row of the trip_id
        cols_agg = ["total_predicted_duration_mins", "total_predicted_distance_miles_for_fare",
                    "origin_metro_area_name", "commute_minutes", "commute_distance", "is_same_day_ride",
                    "trip_starts_during_peak_hours", "dollars_paid_to_driver", "cumulative_boost_amount_cents",
                    "scheduled_starts_at", "boost_ind", "claimed_at", "created_at"]
        self.df_agg = self.df[self.df.boost_ind == 1].groupby("trip_id", as_index=False)[cols_agg].max()

        # Calculate base fare and deviation in percentage from base fare
        self.df_agg["total_boost_dollars"] = self.df_agg.cumulative_boost_amount_cents / 100
        self.df_agg = self.df_agg.drop("cumulative_boost_amount_cents", axis=1)
        self.df_agg["base_fare"] = self.df_agg.apply(lambda x: self.assign_base_fare(x), axis=1)
        self.df_agg["boost_pct_base_fare"] = (self.df_agg.total_boost_dollars / self.df_agg.base_fare).mul(100).round(2)
        # For this model, we only care about trips that received a boost, so we drop it after using

        # Drop one trip with total_boost_dollars greater than dollars_paid_to_driver
        self.df_agg = self.df_agg[self.df_agg.boost_pct_base_fare >= 0]

        # Calculate time fields
        self.df_agg["days_btw_created_and_claimed"] = (
                ((self.df_agg.claimed_at - self.df_agg.created_at) / np.timedelta64(1,'h')) / 24)
        self.df_agg["days_btw_created_and_schedstart"] = (
                ((self.df_agg.scheduled_starts_at - self.df_agg.created_at) / np.timedelta64(1, 'h')) / 24)
        self.df_agg["days_btw_claimed_and_schedstart"] = (
                ((self.df_agg.scheduled_starts_at - self.df_agg.claimed_at) / np.timedelta64(1, 'h')) / 24)
        cols_drop = [
            "boost_ind", "total_boost_dollars", "dollars_paid_to_driver", "base_fare", "claimed_at", "created_at"]
        self.df_agg = self.df_agg.drop(cols_drop, axis=1)


    def feature_engineering(self):
        # Get dummies from metro area
        self.df_agg["origin_metro_area_name"] = self.df_agg.origin_metro_area_name.str.lower().str.replace(" ", "")
        self.df_agg = pd.get_dummies(self.df_agg, dtype=int, prefix="", prefix_sep="")

        # Get sine and cosine transformations
        self.df_agg['sin_month'] = np.sin(2 * np.pi * self.df_agg['scheduled_starts_at'].dt.month / 12)
        self.df_agg['cos_month'] = np.cos(2 * np.pi * self.df_agg['scheduled_starts_at'].dt.month / 12)
        self.df_agg['sin_day_of_week'] = np.sin(2 * np.pi * self.df_agg['scheduled_starts_at'].dt.dayofweek / 7)
        self.df_agg['cos_day_of_week'] = np.cos(2 * np.pi * self.df_agg['scheduled_starts_at'].dt.dayofweek / 7)
        self.df_agg['sin_hour'] = np.sin(2 * np.pi * self.df_agg['scheduled_starts_at'].dt.hour / 24)
        self.df_agg['cos_hour'] = np.cos(2 * np.pi * self.df_agg['scheduled_starts_at'].dt.hour / 24)
        # Drop scheduled_starts_at, we don't need it anymore
        self.df_agg = self.df_agg.drop("scheduled_starts_at", axis=1)

    def write_modeling_data_as_requested_in_instructions(self, path="data/model_df.csv"):
        # The instructions request to save the data for modeling separately. I'll add the code to write it to data/
        self.df_agg.to_csv(path, index=False)


    def prep_data_for_modeling(self):
        self.X = self.df_agg.drop(["boost_pct_base_fare", "trip_id"], axis=1)
        self.y = self.df_agg["boost_pct_base_fare"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.RANDOM_STATE)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        logger.info("\n\nModeling datasets shapes")
        logger.info(f"The shape of X_train is {self.X_train.shape}")
        logger.info(f"The shape of y_train is {self.y_train.shape}")
        logger.info(f"The shape of X_test is {self.X_test.shape}")
        logger.info(f"The shape of y_test is {self.y_test.shape}")

    def train_elastic_net(self):
        self.elastic_net = ElasticNet(random_state=self.RANDOM_STATE)
        param_dist = {
            "alpha": [.1, .2, .3, .4, .5, .6, .7, .8, .9],
            "l1_ratio": [.05, .15, .25, .35, .45, .55, .65, .75, .85, .95]
        }
        random_search = RandomizedSearchCV(
            self.elastic_net, param_distributions=param_dist, n_iter=10, scoring="r2",
            cv=10, random_state=self.RANDOM_STATE, n_jobs=-1
        )
        random_search.fit(self.X_train_scaled, self.y_train)
        self.elastic_net = random_search.best_estimator_

        y_pred = self.elastic_net.predict(self.X_test_scaled)
        self.mae = mean_absolute_error(self.y_test, y_pred)
        self.mse = mean_squared_error(self.y_test, y_pred)
        self.r2 = r2_score(self.y_test, y_pred)

    def report_performance(self):
        logger.info("\n\nElastic net performance...")
        logger.info(f"Mean Absolute Error: {int(self.mae):,}")
        logger.info(f"Mean Squared Error: {int(self.mse):,}")
        logger.info(f"R Score: {self.r2:.2f}")

    def display_model_coefficients(self):
        elastic_net_coeffs = pd.DataFrame({
            "Feature": self.X.columns,
            "Coefficient": self.elastic_net.coef_
        }).sort_values(by="Coefficient", ascending=False)
        logger.info("\n\nElastic net coefficients...")
        logger.info(elastic_net_coeffs)


    def pipeline(self):
        logger.info("\n\nCleaning data...")
        self.clean_data()
        logger.info("\n\nAggregating at the trip level for modeling...")
        self.aggregate_for_analysis()
        logger.info("\n\nEngineering the features...")
        self.feature_engineering()
        logger.info("\n\nPreparing the data for modeling...")
        self.prep_data_for_modeling()
        path = "data/model_df.csv"
        logger.info(f"\n\nWriting modeling data to {path}...")
        self.write_modeling_data_as_requested_in_instructions(path)
        logger.info("\n\nTraining the model...")
        self.train_elastic_net()
        logger.info("\n\nReporting the performance of the model...")
        self.report_performance()
        logger.info("\n\nDisplay Elastic Net coefficients...")
        self.display_model_coefficients()
        logger.info("\n\nDONE...")


if __name__ == "__main__":
    model = ElasticNetDeviationFromBaseFare(boost_df=pd.read_csv("data/boost_df.csv"))
    model.pipeline()