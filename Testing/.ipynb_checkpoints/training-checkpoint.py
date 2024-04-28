# # Suppressing warnings
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

    
# import joblib
# import os
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR
# from xgboost import XGBRegressor
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Load the selected data from a CSV file
# selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

# def train_and_save_models(selected_data, coin_number):
#     selected_data = selected_data.copy()

#     for lag in range(1, 4):
#         selected_data.loc[:, f'{selected_data.columns[coin_number]}_lag_{lag}'] = selected_data[selected_data.columns[coin_number]].shift(lag)

#     selected_data.dropna(inplace=True)

#     features = [f'{selected_data.columns[coin_number]}_lag_{lag}' for lag in range(1, 4)]
#     X = selected_data[features]
#     y = selected_data[selected_data.columns[coin_number]]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Define the Dense layer with a unique name for each model
#     dense_layer = Dense(units=1, name=f"output_layer_{coin_number+1}")

#     models = {
#         'GRADIENT BOOSTING': GradientBoostingRegressor(),
#         'SVR': SVR(),
#         'XGBOOST': XGBRegressor(),
#         'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1))])  # Define LSTM model without Dense layer
#     }

#     for model_name, model in models.items():
#         if model_name == 'LSTM':
#             model.add(dense_layer)  # Add Dense layer to LSTM model
#             model.compile(optimizer='adam', loss='mean_squared_error')
#             X_train_array = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
#             model.fit(X_train_array, y_train, epochs=100, batch_size=32, verbose=0)

#             # Save the trained LSTM model
#             model_filename = f"Model_SELECTED_COIN_{coin_number+1}/{model_name.lower().replace(' ', '_')}_model.h5"
#             os.makedirs(os.path.dirname(model_filename), exist_ok=True)
#             model.save(model_filename)
#             print(f"Trained {model_name} model saved as {model_filename}")
#             continue  # Skip the rest of the loop for LSTM


#         # Train other models
#         model.fit(X_train, y_train)
#         # Save the trained model
#         model_filename = f"Model_SELECTED_COIN_{coin_number+1}/{model_name.lower().replace(' ', '_')}_model.pkl"
#         os.makedirs(os.path.dirname(model_filename), exist_ok=True)
#         joblib.dump(model, model_filename)
#         print(f"Trained {model_name} model saved as {model_filename}")

# # Call the function to train and save all models for each coin
# for coin_number in range(selected_data.shape[1]):
#     train_and_save_models(selected_data, coin_number)

import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the selected data from a CSV file
selected_data = pd.read_csv("Selected_coins.csv", index_col='Date')

def train_and_save_models(selected_data, coin_number):
    selected_data = selected_data.copy()

    for lag in range(1, 4):
        selected_data.loc[:, f'{selected_data.columns[coin_number]}_lag_{lag}'] = selected_data[selected_data.columns[coin_number]].shift(lag)

    selected_data.dropna(inplace=True)

    features = [f'{selected_data.columns[coin_number]}_lag_{lag}' for lag in range(1, 4)]
    X = selected_data[features]
    y = selected_data[selected_data.columns[coin_number]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Gradient Boosting': GradientBoostingRegressor(),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(),
        'LSTM': Sequential([LSTM(units=50, input_shape=(X_train.shape[1], 1))])  # Define LSTM model without Dense layer
    }

    # Hyperparameter grid for grid search
    param_grid = {
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 0.9]
        },
        'SVR': {'C': [1, 10], 'gamma': ['scale', 'auto']},
        'XGBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.01],
            'max_depth': [3, 5],
            'subsample': [0.8, 0.9]
        },
    }

    for model_name, model in models.items():
        if model_name == 'LSTM':
            model.add(Dense(units=1))  # Add Dense layer to LSTM model
            model.compile(optimizer='adam', loss='mean_squared_error')
            X_train_array = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
            model.fit(X_train_array, y_train, epochs=100, batch_size=32, verbose=0)

            # Save the trained LSTM model
            model_filename = f"Model_SELECTED_COIN_{coin_number+1}/lstm_model.h5"
            os.makedirs(os.path.dirname(model_filename), exist_ok=True)
            model.save(model_filename)
            print(f"Trained {model_name} model saved as {model_filename}")
            continue  # Skip the rest of the loop for LSTM

        # Perform grid search for hyperparameter tuning
        params = param_grid[model_name]
        model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error')
        model.fit(X_train, y_train)
        model = model.best_estimator_

        # Train other models
        model.fit(X_train, y_train)

        # Save the trained model
        model_filename = f"Model_SELECTED_COIN_{coin_number+1}/{model_name.lower().replace(' ', '_')}_model.pkl"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Trained {model_name} model saved as {model_filename}")

# Call the function to train and save all models for each coin
for coin_number in range(selected_data.shape[1]):
    train_and_save_models(selected_data, coin_number)

