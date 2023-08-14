
# import gradio as gr
# from gradio.components import Number, Dropdown
# import numpy as np
# import pandas as pd
# import pickle

# # Load exported data
# exported_data_path = 'src\Asset\ML\my_exported_components.pkl'
# with open(exported_data_path, 'rb') as file:
#     exported_data = pickle.load(file)

# # Load the exported components
# categorical_imputer = exported_data['categorical_imputer']
# numerical_imputer = exported_data['numerical_imputer']
# encoder = exported_data['encoder']
# scaler = exported_data['scaler']
# best_model = exported_data['best_model']

# # # Define the function to preprocess and predict
# # def churn_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
# #                      InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
# #                      StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
# #                      MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges):
# #     # Preprocess the input data
# #     input_data = [[gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
# #                    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
# #                    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
# #                    MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges]]

# #     input_data = categorical_imputer.transform(input_data)
# #     input_data = numerical_imputer.transform(input_data)
# #     input_data = encoder.transform(input_data)
# #     input_data = scaler.transform(input_data)

# #     # Make predictions using the loaded model
# #     prediction = best_model.predict(input_data)

# #     return prediction

# def churn_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
#                      InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#                      StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
#                      MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges):
    
#     # Categorical features
#     categorical_features = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, 
#                             InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#                             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]
    
#     # Numerical features
#     numerical_features = [tenure, MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges]

#     # Preprocess categorical features
#     categorical_input = [categorical_features]
#     categorical_input = categorical_imputer.transform(categorical_input)
#     categorical_input = encoder.transform(categorical_input)

#     # Preprocess numerical features
#     numerical_input = [numerical_features]
#     numerical_input = numerical_imputer.transform(numerical_input)
#     numerical_input = scaler.transform(numerical_input)

#     # Combine preprocessed categorical and numerical features
#     # input_data = np.concatenate((categorical_input, numerical_input), axis=1)
#     input_data = np.hstack((categorical_input, numerical_input))


#     # Make predictions using the loaded model
#     prediction = best_model.predict(input_data)


#     return prediction

# # Define input components
# input_components = [
#     gr.inputs.Dropdown(choices=['Female', 'Male'], label='gender'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='SeniorCitizen'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='Partner'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='Dependents'),
#     gr.inputs.Number(label='tenure'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='PhoneService'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='MultipleLines'),
#     gr.inputs.Dropdown(choices=['DSL', 'Fiber optic', 'No'], label='InternetService'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='OnlineSecurity'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='OnlineBackup'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='DeviceProtection'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='TechSupport'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='StreamingTV'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='StreamingMovies'),
#     gr.inputs.Dropdown(choices=['Month-to-month', 'One year', 'Two year'], label='Contract'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='PaperlessBilling'),
#     gr.inputs.Dropdown(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='PaymentMethod'),
#     gr.inputs.Number(label='MonthlyCharges'),
#     gr.inputs.Number(label='TotalCharges'),
#     gr.inputs.Number(label='MonthlyCharges_TotalCharges_Ratio'),
#     gr.inputs.Number(label='AverageMonthlyCharges')
# ]

# # Create and launch the Gradio interface
# # iface = gr.Interface(fn=churn_prediction, inputs=input_components, outputs="text")
# # iface.launch()
# try:
#     # Create and launch the Gradio interface
#     iface = gr.Interface(fn=churn_prediction, inputs=input_components, outputs="text")
#     iface.launch()
# except Exception as e:
#     print("An error occurred:", e)



# import gradio as gr
# from gradio.components import Number, Dropdown
# import pickle
# import numpy as np

# # Load exported data
# exported_data_path = 'src\Asset\ML\my_exported_components.pkl'
# with open(exported_data_path, 'rb') as file:
#     exported_data = pickle.load(file)

# # Load the exported components
# categorical_imputer = exported_data['categorical_imputer']
# numerical_imputer = exported_data['numerical_imputer']
# encoder = exported_data['encoder']
# scaler = exported_data['scaler']
# best_model = exported_data['best_model']

# Define the function to preprocess and predict
# def churn_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
#                      InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#                      StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
#                      MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges):
#     # Categorical features
#     categorical_features = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, 
#                             InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#                             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]
    
#     # Numerical features
#     numerical_features = [tenure, MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges]

#     # Preprocess categorical features
#     categorical_input = [categorical_features]
#     categorical_input = categorical_imputer.transform(categorical_input)
#     categorical_input = encoder.transform(categorical_input)

#     # Preprocess numerical features
#     numerical_input = [numerical_features]
#     numerical_input = numerical_imputer.transform(numerical_input)
#     numerical_input = scaler.transform(numerical_input)

#     # Combine preprocessed categorical and numerical features
#     # input_data = np.concatenate((categorical_input, numerical_input), axis=1)
#     input_data = np.hstack((categorical_input, numerical_input))


#     # Make predictions using the loaded model
#     prediction = best_model.predict(input_data)


#     return prediction
# def churn_prediction(inputs):
#     # Categorical features
#     categorical_features = [inputs['gender'], inputs['SeniorCitizen'], inputs['Partner'], 
#                             inputs['Dependents'], inputs['PhoneService'], inputs['MultipleLines'], 
#                             inputs['InternetService'], inputs['OnlineSecurity'], inputs['OnlineBackup'], 
#                             inputs['DeviceProtection'], inputs['TechSupport'], inputs['StreamingTV'], 
#                             inputs['StreamingMovies'], inputs['Contract'], inputs['PaperlessBilling'], 
#                             inputs['PaymentMethod']]
    
#     # Numerical features
#     numerical_features = [inputs['tenure'], inputs['MonthlyCharges'], inputs['TotalCharges'],
#                           inputs['MonthlyCharges_TotalCharges_Ratio'], inputs['AverageMonthlyCharges']]

#     # Preprocess categorical features
#     categorical_input = [categorical_features]
#     categorical_input = categorical_imputer.transform(categorical_input)
#     categorical_input = encoder.transform(categorical_input)

#     # Preprocess numerical features
#     numerical_input = [numerical_features]
#     numerical_input = numerical_imputer.transform(numerical_input)
#     numerical_input = scaler.transform(numerical_input)

#     # Combine preprocessed categorical and numerical features
#     input_data = np.hstack((categorical_input, numerical_input))

#     # Make predictions using the loaded model
#     prediction = best_model.predict(input_data)

#     return prediction

# # Define input components
# input_components = [
#     gr.inputs.Dropdown(choices=['Female', 'Male'], label='gender'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='SeniorCitizen'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='Partner'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='Dependents'),
#     gr.inputs.Number(label='tenure'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='PhoneService'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='MultipleLines'),
#     gr.inputs.Dropdown(choices=['DSL', 'Fiber optic', 'No'], label='InternetService'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='OnlineSecurity'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='OnlineBackup'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='DeviceProtection'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='TechSupport'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='StreamingTV'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='StreamingMovies'),
#     gr.inputs.Dropdown(choices=['Month-to-month', 'One year', 'Two year'], label='Contract'),
#     gr.inputs.Dropdown(choices=['No', 'Yes'], label='PaperlessBilling'),
#     gr.inputs.Dropdown(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='PaymentMethod'),
#     gr.inputs.Number(label='MonthlyCharges'),
#     gr.inputs.Number(label='TotalCharges'),
#     gr.inputs.Number(label='MonthlyCharges_TotalCharges_Ratio'),
#     gr.inputs.Number(label='AverageMonthlyCharges')
# ]

# # Create and launch the Gradio interface
# iface = gr.Interface(fn=churn_prediction, inputs=input_components, outputs="text")
# iface.launch()



# import gradio as gr
# from gradio.components import Dropdown, Number, Label
# import pickle
# import numpy as np

# # Load exported data
# exported_data_path = 'src\Asset\ML\my_exported_components.pkl'
# with open(exported_data_path, 'rb') as file:
#     exported_data = pickle.load(file)

# # Load the exported components
# categorical_imputer = exported_data['categorical_imputer']
# numerical_imputer = exported_data['numerical_imputer']
# encoder = exported_data['encoder']
# scaler = exported_data['scaler']
# best_model = exported_data['best_model']

# # Define the function to preprocess and predict
# def churn_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,
#                      InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#                      StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
#                      MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges):
#     # Categorical features
#     categorical_features = [gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, 
#                             InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
#                             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]
    
#     # Numerical features
#     numerical_features = [tenure, MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges]

#     # Preprocess categorical features
#     categorical_input = [categorical_features]
#     categorical_input = categorical_imputer.transform(categorical_input)
#     categorical_input = encoder.transform(categorical_input)

#     # Preprocess numerical features
#     numerical_input = [numerical_features]
#     numerical_input = numerical_imputer.transform(numerical_input)
#     numerical_input = scaler.transform(numerical_input)

#     # Combine preprocessed categorical and numerical features
#     input_data = np.hstack((categorical_input, numerical_input))

#     # Make predictions using the loaded model
#     prediction = best_model.predict(input_data)

#     return prediction

# # Define input components
# input_components = [
#     Dropdown(choices=['Female', 'Male'], label='gender'),
#     Dropdown(choices=['No', 'Yes'], label='SeniorCitizen'),
#     Dropdown(choices=['No', 'Yes'], label='Partner'),
#     Dropdown(choices=['No', 'Yes'], label='Dependents'),
#     Number(label='tenure'),
#     Dropdown(choices=['No', 'Yes'], label='PhoneService'),
#     Dropdown(choices=['No', 'Yes'], label='MultipleLines'),
#     Dropdown(choices=['DSL', 'Fiber optic', 'No'], label='InternetService'),
#     Dropdown(choices=['No', 'Yes'], label='OnlineSecurity'),
#     Dropdown(choices=['No', 'Yes'], label='OnlineBackup'),
#     Dropdown(choices=['No', 'Yes'], label='DeviceProtection'),
#     Dropdown(choices=['No', 'Yes'], label='TechSupport'),
#     Dropdown(choices=['No', 'Yes'], label='StreamingTV'),
#     Dropdown(choices=['No', 'Yes'], label='StreamingMovies'),
#     Dropdown(choices=['Month-to-month', 'One year', 'Two year'], label='Contract'),
#     Dropdown(choices=['No', 'Yes'], label='PaperlessBilling'),
#     Dropdown(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='PaymentMethod'),
#     Number(label='MonthlyCharges'),
#     Number(label='TotalCharges'),
#     Number(label='MonthlyCharges_TotalCharges_Ratio'),
#     Number(label='AverageMonthlyCharges')
# ]

# # Create and launch the Gradio interface
# iface = gr.Interface(fn=churn_prediction, inputs=input_components, outputs="text")
# iface.launch()


import gradio as gr
from gradio.components import Number, Dropdown
import numpy as np
import pandas as pd
import pickle
# from sklearn.metrics import PredictionErrorDisplay

# Load exported data
exported_data_path = 'src\Asset\ML\my_exported_components.pkl'
with open(exported_data_path, 'rb') as file:
    exported_data = pickle.load(file)

# Load the exported components
categorical_imputer = exported_data['categorical_imputer']
numerical_imputer = exported_data['numerical_imputer']
encoder = exported_data['encoder']
scaler = exported_data['scaler']
best_model = exported_data['best_model']

def churn_prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
                     InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                     StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                     MonthlyCharges, TotalCharges, MonthlyCharges_TotalCharges_Ratio, AverageMonthlyCharges):
    
    # Create a DataFrame with the provided inputs
    prediction_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges],
        'MonthlyCharges_TotalCharges_Ratio': [MonthlyCharges_TotalCharges_Ratio],
        'AverageMonthlyCharges': [AverageMonthlyCharges]
    })
    
    # Preprocessing for categorical data
    prediction_data_categorical = prediction_data.select_dtypes(include='object')
    prediction_data_encoded = encoder.transform(categorical_imputer.transform(prediction_data_categorical))

    # Convert the encoded sparse matrix to a DataFrame
    prediction_data_encoded_df = pd.DataFrame.sparse.from_spmatrix(prediction_data_encoded,
                                                                  columns=encoder.get_feature_names_out(prediction_data_categorical.columns),
                                                                  index=prediction_data_categorical.index)

    # Preprocessing for numerical data
    prediction_data_numerical = prediction_data.select_dtypes(include=['int', 'float'])
    prediction_data_scaled = scaler.transform(numerical_imputer.transform(prediction_data_numerical))

    # Convert the scaled numerical data to a DataFrame
    prediction_data_scaled_df = pd.DataFrame(prediction_data_scaled,
                                             columns=prediction_data_numerical.columns,
                                             index=prediction_data_numerical.index)

    # Concatenate the encoded categorical data and scaled numerical data
    prediction_data_preprocessed = pd.concat([prediction_data_encoded_df, prediction_data_scaled_df], axis=1)

    # Make predictions using the loaded model
    predictions = best_model.predict(prediction_data_preprocessed)

     # Map the predictions to 'Yes' or 'No'
    prediction_label = ['Customer Churn' if p == 1 else 'Customer Not Churn' for p in predictions]
    
    return prediction_label


# Define input components
input_components = [
    gr.inputs.Dropdown(choices=['Female', 'Male'], label='gender'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='SeniorCitizen'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='Partner'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='Dependents'),
    gr.inputs.Number(label='tenure'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='PhoneService'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='MultipleLines'),
    gr.inputs.Dropdown(choices=['DSL', 'Fiber optic', 'No'], label='InternetService'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='OnlineSecurity'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='OnlineBackup'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='DeviceProtection'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='TechSupport'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='StreamingTV'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='StreamingMovies'),
    gr.inputs.Dropdown(choices=['Month-to-month', 'One year', 'Two year'], label='Contract'),
    gr.inputs.Dropdown(choices=['No', 'Yes'], label='PaperlessBilling'),
    gr.inputs.Dropdown(choices=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label='PaymentMethod'),
    gr.inputs.Number(label='MonthlyCharges'),
    gr.inputs.Number(label='TotalCharges'),
    gr.inputs.Number(label='MonthlyCharges_TotalCharges_Ratio'),
    gr.inputs.Number(label='AverageMonthlyCharges')
]

# Create and launch the Gradio interface
iface = gr.Interface(fn=churn_prediction, inputs=input_components, outputs="text")
iface.launch(share=True)


# , 'No internet service'