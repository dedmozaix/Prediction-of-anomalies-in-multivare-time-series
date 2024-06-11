This paper is the author's final qualification work on the topic "Prediction of anomalies in multivariate time series". The code is a system of data prediction by Multivariant ARIMA, LSTM and DSANet methods with subsequent anomaly detection by DeepSVDD and COPOD methods. 

To run the program, install the repository, and upload your data to a database (mine is Postgres). You need to import into your main the classes from three files: BaseObjects, AbstractTrainer and Hyperparams. 
BaseObjects stores uninitialized class objects of prediction models and simple database connection configuration, which can be easily modified as you see fit. 
The HparamsDefiner class of the Hyperparams file operates on model parameters that can be modified.
The main class is the AbstractModelTrainer class of the AbstractTrainer file. It initializes objects of prediction and anomaly detection model classes in itself, and with the help of pytorch_lightning it automatically feeds the training data. 

An example of use is given in the main.py file



Translated with www.DeepL.com/Translator (free version)
