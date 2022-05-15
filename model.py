import joblib
import numpy as np
import glob
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from helper_functions import *

vehicles = glob.glob('C://Users//John Emad//Desktop//imageprocessing//Data//vehicles//*//*.png')
non_vehicles = glob.glob('C://Users//John Emad//Desktop//imageprocessing//Data//non-vehicles//*//*.png')

def prepare_dataset(vehicles,non_vehicles):

    vehicle_features = extract_features(vehicles, cspace='YUV')
    non_vehicle_features = extract_features(non_vehicles, cspace='YUV')
    labels = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))
    scaled_X, X_scaler = plot_features(vehicle_features, non_vehicle_features)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, labels, test_size=0.2, random_state=4000)
    return X_train, X_test, y_train, y_test,scaled_X, X_scaler

if __name__ == '__main__':
    X_train, X_test, y_train, y_test,scaled_X, X_scaler = prepare_dataset(vehicles,non_vehicles)
    ##################################### Using SVM Classifier ######################################
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    print('SVC results')
    print('accuracy on training data: ', svc.score(X_train, y_train))
    print('accuracy on test data: ', svc.score(X_test, y_test))

    ##################################### Using MLP Classifier ######################################

    mlp = MLPClassifier(random_state=999)
    mlp.fit(X_train, y_train)
    print('MLP results')
    print('accuracy on training data: ', mlp.score(X_train, y_train))
    print('accuracy on test data: ', mlp.score(X_test, y_test))
    prediction = mlp.predict(X_test[0].reshape(1, -1))

    ##################################### Loading MLP data ######################################

    joblib.dump(mlp, 'mlp1.pkl')
    joblib.dump(X_scaler, 'scaler1.pkl')
