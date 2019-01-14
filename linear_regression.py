import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

def getData():
    # Get home data from CSV file
    dataFile = pd.read_csv('home_price_data.csv')
    return dataFile

def linearRegressionModel(X_train, Y_train, X_test, Y_test):
    linear = linear_model.LinearRegression()

    # Training process
    linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = linear.score(X_test, Y_test)
    y_pre = linear.predict(X_test)
    print('gia du doan: ',int(y_pre[0]))
    print('gia thuc te:',int(Y_test[0]))
    print('He so dung thu vien tinh:', linear.coef_)
    return score_trained

def find_coefficient(X_train, Y_train, X_test,Y_test):
    linear = np.dot(np.linalg.pinv(np.dot(X_train.T,X_train)),np.dot(X_train.T, Y_train))
    return linear

if __name__ == "__main__":
    data = getData()
    if data is not None:
        # Selection few attributes
        attributes = list(
            ['num_bed',
            'year_built',
            'longitude',
            'latitude',
            'num_room',
            'num_bath',
            'living_area',
            'property_type',
            'num_parking',
            'accessible_buildings',
            'family_quality',
            'art_expos',
            'emergency_shelters',
            'emergency_water',
            'Facilities',
            'fire_stations',
            'Cultural',
            'Monuments',
            'police_stations',
            'Vacant',
            'Free_Parking']
        )
        # Vector attributes of house
        X = data[attributes]
        # Vector price of house
        Y = data['askprice']
        
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)
        
        # Linear Regression Model
        linearScore = linearRegressionModel(X_train, Y_train, X_test, Y_test)
        print ('Linear Score = ' , linearScore)

        w=find_coefficient(X_train, Y_train, X_test, Y_test)
        print('He so tu tinh:', w.T)