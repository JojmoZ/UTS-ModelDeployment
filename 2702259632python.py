
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import pickle

class mymodel:
    def __init__(self, csv):
        self.csv = csv
        self.df = pd.read_csv(csv)
        self.model = None

    def preprocess(self):
        self.df['booking_status'] = self.df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})
        self.df.drop('Booking_ID', axis=1, inplace=True)

        numer_imputer = SimpleImputer(strategy='median')
        self.df[['required_car_parking_space', 'avg_price_per_room']] = numer_imputer.fit_transform(
            self.df[['required_car_parking_space', 'avg_price_per_room']])

        categ_imputer = SimpleImputer(strategy='most_frequent')
        self.df[['type_of_meal_plan']] = categ_imputer.fit_transform(self.df[['type_of_meal_plan']])

        categ_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        self.label_encoders = {}
        for col in categ_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

    def train_and_select_model(self):
        X = self.df.drop('booking_status', axis=1)
        y = self.df['booking_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=241)

        rf = RandomForestClassifier(random_state=241)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_report = classification_report(y_test, rf_pred, output_dict=True)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=241)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_report = classification_report(y_test, xgb_pred, output_dict=True)

        if rf_report['1']['f1-score'] > xgb_report['1']['f1-score']:
            self.model = rf
            print("Random Forest selected.")
        else:
            self.model = xgb
            print("XGBoost selected.")

        with open('best_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved to best_model.pkl")

if __name__ == "__main__":
    model = mymodel('Dataset_B_hotel.csv')
    model.preprocess()
    model.train_and_select_model()
