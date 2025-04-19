from django.shortcuts import render
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
import tensorflow as tf
import seaborn as sns
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import random
import io
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

random.seed(42)
# Create your views here.


def index(request):
    return render(request,'myapp/index.html')



def login(request):

    if request.method == "POST":
        username = request.POST['uname']
        password = request.POST['pwd']
        print(username, password)
        if username == 'admin' and password == 'admin':

            return render(request, 'myapp/homepage.html')
    return render(request,'myapp/login.html')



def homepage(request):
    return render(request,'myapp/homepage.html')



def dataupload(request):
    X_train_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/X_train_G_OR_R.csv')
    X_test_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/X_test_G_OR_R.csv')
    y_train_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/y_train_G_OR_R.csv')
    y_test_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/y_test_G_OR_R.csv')

    print(X_train_label_G_OR_B.head())
    content = {
        'data': X_train_label_G_OR_B.shape[0],
        'tcol': X_train_label_G_OR_B.shape[1]

    }
    # Gather train and test data counts
    train_count = len(y_train_label_G_OR_B)
    test_count = len(y_test_label_G_OR_B)

    plt.figure(figsize=(8, 4))
    bars = plt.bar(['Train', 'Test'], [train_count, test_count], color=['blue', 'orange'])
    plt.xlabel('Dataset')
    plt.ylabel('Data Count')
    plt.title('Train and Test Data Count')

    # Display count numbers on top of the bars
    for bar, count in zip(bars, [train_count, test_count]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, count, ha='center', va='bottom')

    # plt.show()
    return render(request,'myapp/dataupload.html',content)



def modelcreation(request):

    X_train_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/X_train_G_OR_R.csv')
    X_test_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/X_test_G_OR_R.csv')
    y_train_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/y_train_G_OR_R.csv')
    y_test_label_G_OR_B = pd.read_csv('Good_or_Ransomware_model_data/y_test_G_OR_R.csv')
    # Set random seed for numpy (for data shuffling, etc.)
    np.random.seed(42)

    # Set random seed for Python's built-in random module (if used)

    # Set random seed for TensorFlow
    tf.random.set_seed(42)
    # Scale the features using StandardScaler
    scaler = StandardScaler()

    X_train_label = scaler.fit_transform(X_train_label_G_OR_B)
    X_test_label = scaler.transform(X_test_label_G_OR_B)

    # Reshape the data for LSTM input [samples, timesteps, features]
    X_train_label = np.reshape(X_train_label, (X_train_label.shape[0], 1, X_train_label.shape[1]))
    X_test_label = np.reshape(X_test_label, (X_test_label.shape[0], 1, X_test_label.shape[1]))

    # Build the LSTM model
    G_or_R_LSTM_model = Sequential()
    G_or_R_LSTM_model.add(LSTM(6, input_shape=(X_train_label.shape[1], X_train_label.shape[2])))
    G_or_R_LSTM_model.add(Dense(1, activation='sigmoid'))  # Binary classification output

    # Compile the model
    G_or_R_LSTM_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(G_or_R_LSTM_model.summary())
    # Model summary
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Print the model summary to the redirected stdout
    # model_lstm.summary()
    G_or_R_LSTM_model.summary()
    # Get the model summary as a string
    summary_string = sys.stdout.getvalue()

    # Reset stdout to its original value
    sys.stdout = original_stdout

    # Now, `summary_string` contains the model summary
    print(summary_string)
    content1 = {
        'data': summary_string
    }
    # Train the model with early stopping
    # history = G_or_R_LSTM_model.fit(X_train_label, y_train_label_G_OR_B, epochs=30, batch_size=12,
    #                                 validation_data=(X_test_label, y_test_label_G_OR_B))
    #
    # # Predict using the trained LSTM model
    # predictions = G_or_R_LSTM_model.predict(X_test_label)
    #
    # # Convert predictions to binary using a threshold (e.g., 0.5)
    # print(predictions[:10])
    # binary_predictions = (predictions > 0.5).astype(int)
    # print(binary_predictions[:10])
    #
    # predictions = G_or_R_LSTM_model.predict(X_test_label)
    #
    # binary_predictions = (predictions > 0.5).astype(int)
    return render(request,'myapp/modelcreation.html',content1)



def randompredict(request):

    if request.method=="POST":
        # Load the saved LSTM model
        model = load_model('C:/Users/user/Documents/project/ransomeware/model/G_or_R_LSTM_model.h5')

        # Load new data for prediction (modify this part according to your actual data)
        new_data = pd.read_csv('C:/Users/user/Documents/project/ransomeware/Good_or_Ransomware_model_data/X_test_G_OR_R.csv')

        # Assuming 'new_data' is in the same format as your training data
        # Perform the same preprocessing as you did for training data
        scaler = StandardScaler()

        new_data_scaled = scaler.fit_transform(new_data)
        new_data_reshaped = np.reshape(new_data_scaled, (new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))

        # Make predictions
        predictions = model.predict(new_data_reshaped)

        # Convert predictions to binary using a threshold (e.g., 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        # binary_predictions=gb
        print(binary_predictions)
        # Map the output (Ransomware=1, Goodware=0)
        mapped_predictions = np.where(binary_predictions == 1, 'Ransomware', 'Goodware')
        print(mapped_predictions)
        # Display the xtest values and predictions for each row
        result_df = pd.DataFrame(data=np.concatenate([new_data.values, mapped_predictions.reshape(-1, 1)], axis=1),
                                 columns=list(new_data.columns) + ['Predictions'])

        # Save the results to a CSV file
        result_df.to_csv('C:/Users/user/Documents/project/ransomeware/Good_or_Ransomware_model_data/G_or_B_prediction_results.csv', index=False)
        inputdf=pd.read_csv('C:/Users/user/Documents/project/ransomeware/Good_or_Ransomware_model_data/G_or_B_prediction_results.csv')
        # Display the results
        print("Predictions:")
        print(result_df)
        # Display the results
        # print("Predictions:")
        print(result_df[['Predictions']].head(1))
        res = result_df[['Predictions']]
        print(res)
        with open('C:/Users/user/Documents/project/ransomeware/Good_or_Ransomware_model_data/G_or_B_prediction_results.csv', 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = list(csv_reader)
        return render(request, 'myapp/randompredict.html',{'data':data})
    return render(request,'myapp/randompredict.html')



def rfamilypredict(request):
    if request.method == 'POST':
        # Load the saved LSTM model
        model = load_model('model/Family_multiclass_lstm_model.h5')

        # Load new data for prediction (modify this part according to your actual data)
        new_data = pd.read_csv(
            'Ransomware_Family_model_data/X_test_Family.csv')  # Replace with the path to your new data

        # Assuming 'new_data' is in the same format as your training data
        # Perform the same preprocessing as you did for training data
        scaler = StandardScaler()
        new_data_scaled = scaler.fit_transform(new_data.values.reshape(-1, 1)).reshape(new_data.shape[0],
                                                                                       new_data.shape[1], 1)

        # Make predictions
        predictions = model.predict(new_data_scaled)

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Define the mapping between IDs and Family Information
        id_to_family = {
            0: {'Type': 'Goodware',
                'Prevention': 'Ensure downloading software only from trusted sources. Regularly update and patch all software to prevent vulnerabilities.'},
            1: {'Type': 'Critroni',
                'Prevention': 'Regularly back up data, use reliable security software, and avoid clicking on suspicious links or downloading attachments from unknown sources.'},
            2: {'Type': 'CryptLocker',
                'Prevention': 'Keep software and operating systems updated, use strong passwords, and avoid opening suspicious email attachments.'},
            3: {'Type': 'CryptoWall',
                'Prevention': 'Regularly update operating systems and software, use reputable security software, and educate users about phishing emails.'},
            4: {'Type': 'KOLLAH',
                'Prevention': 'Regularly update Linux systems, employ firewall protection, and restrict user permissions.'},
            5: {'Type': 'Kovter',
                'Prevention': 'Use reputable security software, avoid clicking on suspicious links or ads, and regularly update systems.'},
            6: {'Type': 'Locker',
                'Prevention': 'Regularly back up data, employ strong authentication measures, and educate users about phishing attacks.'},
            7: {'Type': 'MATSNU',
                'Prevention': 'Update Windows systems regularly, use strong antivirus software, and be cautious of suspicious emails.'},
            8: {'Type': 'PGPCODER', 'Prevention': 'Regularly back up data, use strong passwords, and update software.'},
            9: {'Type': 'Reveton',
                'Prevention': 'Employ strong security software, do not pay ransoms, and seek professional help if infected.'},
            10: {'Type': 'TeslaCrypt',
                 'Prevention': 'Regularly update systems and applications, use reputable security software, and avoid suspicious websites.'},
            11: {'Type': 'Trojan-Ransom',
                 'Prevention': 'Regularly update antivirus software, avoid suspicious links, and use firewalls.'}
        }

        # Map predicted labels to their corresponding family information
        predicted_family_info = [id_to_family[label] for label in predicted_labels]

        # Display the predictions
        result_df = pd.DataFrame(data=np.concatenate([new_data.values, predicted_labels.reshape(-1, 1)], axis=1),
                                 columns=list(new_data.columns) + ['Predicted_Class'])

        # Add columns for predicted family information
        result_df[['Type', 'Prevention']] = pd.DataFrame(
            [[info['Type'], info['Prevention']] for info in predicted_family_info])

        # Save the results to a CSV file
        result_df.to_csv('Family_prediction_results.csv', index=False)

        # Display the results
        # print("Predictions:")
        # print(result_df)
        # print(predicted_labels)
        result_df[['Predicted_Class', 'Type', 'Prevention']].to_csv('Family_prediction_results123.csv', index=False)

        # Display the results
        print("Predictions:")
        print(result_df[['Predicted_Class', 'Type', 'Prevention']].head(1))
        res = result_df[['Prevention']]
        print(res)
        first_row = result_df.iloc[0]
        print(first_row['Predicted_Class'], first_row['Type'])
        print(first_row['Type'])
        res = first_row['Prevention']
        with open('Family_prediction_results123.csv','r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            data = list(csv_reader)
        return render(request, 'myapp/rfamilypredict.html', {'data':data})
    return render(request,'myapp/rfamilypredict.html')



def viewgraph(request):
    return render(request,'myapp/viewgraph.html')