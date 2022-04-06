# Jonathan Oakey


class Predict_Emissions_From_Engine_Size:
  # Import required libraries:
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn import linear_model

  def __init__(self):
    # Read the CSV file :
    data = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")
    train_testing_data = self.generate_training_testing_data(data)
    train = train_testing_data['train']
    test = train_testing_data['test']
    regr = self.create_and_plot_model(train)


    while True:
      print("")
      print("what would you like to do?")
      print('Choose from one of the choices below:')
      print('Check Accuracy')
      print('Make Predictions')
      print('Explore Data')
      print('Exit')
      the_response = input()
      if 'Exit' in the_response:
        break
      elif 'Check Accuracy' in the_response:
        checking_accuracy(regr,test)
      elif 'Make Predictions' in the_response:
        making_predictions(regr)
      elif 'Explore Data' in the_response:
        explore_data(data)
      else:
        print('unknown response...try again')

  def explore_data(self,data):
    # Let's select some features to explore more :
    data = data[["ENGINESIZE","CO2EMISSIONS"]]
    # ENGINESIZE vs CO2EMISSIONS:
    plt.scatter(data["ENGINESIZE"] , data["CO2EMISSIONS"] , color="blue")
    plt.xlabel("ENGINESIZE")
    plt.ylabel("CO2EMISSIONS")
    plt.show()

  def generate_training_testing_data(self,data):
    train_testing_data = {}
    # Generating training and testing data from our data:
    # We are using 80% data for training.
    train = data[:(int((len(data)*0.8)))]
    test = data[(int((len(data)*0.8))):]
    train_testing_data['train'] = train
    train_testing_data['test'] = test
    return train_testing_data



  def create_and_plot_model(self,train):
    model_data = {}
    # Modeling:
    # Using sklearn package to model data :
    regr = linear_model.LinearRegression()
    train_x = np.array(train[["ENGINESIZE"]])
    train_y = np.array(train[["CO2EMISSIONS"]])
    regr.fit(train_x,train_y)
    # The coefficients:
    print('Model Created')
    print ("coefficients : ",regr.coef_) #Slope
    print ("Intercept : ",regr.intercept_) #Intercept
    # Plotting the regression line:
    plt.scatter(train["ENGINESIZE"], train["CO2EMISSIONS"], color='blue')
    plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    return regr



  def get_regression_predictions(self,input_features,intercept,slope):
    # Predicting values:
    # Function for predicting future values :
    predicted_values = input_features*slope + intercept
    return predicted_values

  def making_predictions(regr):
    # Predicting emission for future car:
    # my_engine_size = 3.5
    print('enter an engine size (ex: 3.5)')
    my_engine_size = input()
    my_engine_size = float(my_engine_size)
    estimatd_emission = get_regression_predictions(my_engine_size,regr.intercept_[0],regr.coef_[0][0])
    print ("Estimated Emission :",estimatd_emission)


  def checking_accuracy(self,regr,test):
    # Checking various accuracy:
    from sklearn.metrics import r2_score
    test_x = np.array(test[['ENGINESIZE']])
    test_y = np.array(test[['CO2EMISSIONS']])
    test_y_ = regr.predict(test_x)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_ , test_y) )




Predict_Emissions_From_Engine_Size()

