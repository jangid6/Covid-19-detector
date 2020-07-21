import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data,ratio):
    #If you set the np.random.seed(a_fixed_number) every time you call the numpy's other random function, the result will be the same:
    np.random.seed(45)
    #generate list of random no's , range -> len(data)    , size -> len(Data)
    shuffled_array = np.random.permutation(len(data))  
    test_array_size = int(len(data)*ratio)
    test_indices = shuffled_array[:test_array_size]
    train_indices = shuffled_array[test_array_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('C:/Users/mohan/OneDrive/Desktop/projects/covid19.csv')
    train_data,test_data = data_split(df,0.2)
    x_train = train_data[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()
    x_test = test_data[['fever','bodyPain','age','runnyNose','diffBreath']].to_numpy()

    #sklearn says,y_train should be in size of train data & commna->.reshape(2400,)
    y_train = train_data[['infectionProb']].to_numpy().reshape(len(train_data),)
    y_test = test_data[['infectionProb']].to_numpy().reshape(len(test_data),)

    clf = LogisticRegression()
    clf.fit(x_train,y_train)

    file_Name = "model.pkl"
    # open the file for writing
    fileObject = open(file_Name,'wb')
    # this writes the object model(clf) to the 'model.pkl' file
    pickle.dump(clf,fileObject)   

    # here we close the fileObject
    fileObject.close()

    
    

