import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class UserPredictor:
    def __init__(self):
        continuous = ["age", "seconds", "past_purchase_amt"]
        discrete = ["badge"]

        imputer_c = SimpleImputer(strategy="mean")
        transformer_c = StandardScaler()
        imputer_d = SimpleImputer(strategy="constant", fill_value="None")
        transformer_d = OneHotEncoder()

        steps_c = [('imputer_c', imputer_c), ('transformer_c', transformer_c)]
        steps_d = [('imputer_d', imputer_d), ('transformer_d', transformer_d)]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('continuous', Pipeline(steps_c), continuous),
                ('discrete', Pipeline(steps_d), discrete)
            ])

        self.model = Pipeline(steps=[('preprocessor', preprocessor), ('clf', RandomForestClassifier())])

    def _process_logs(self, users, logs):
        merged_data = pd.merge(users, logs, on='user_id', how='left')
        
        user_summary = merged_data.groupby(['user_id'], as_index=False).agg({'badge': 'first', 'age': 'first', 'past_purchase_amt': 'first', 'seconds': 'sum'})

        return user_summary

    def fit(self, train_users, train_logs, train_y):

        merged_data = self._process_logs(train_users,train_logs)
        merged_data = pd.merge(merged_data, train_y, on='user_id', how='inner')

        self.model.fit(merged_data[['age', 'seconds', 'past_purchase_amt', 'badge']], merged_data['y'])

    def predict(self, test_users, test_logs):

        processed_logs = self._process_logs(test_users,test_logs)

        return self.model.predict(processed_logs[['age', 'seconds', 'past_purchase_amt', 'badge']])
