# Project 1 : Movie Rating Prediction with Python, CodSoft Internship.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Movie%20Rating%20Prediction%20with%20Python%20Final.ipynb)
## Description : 
The goal is to analyze historical movie data and develop a model that accurately estimates the rating given to a movie by users or critics.
I have used **Linear Regression** and **Random Forest Regression** for building movie rating prediction model.
## Dataset:
[https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies]

# Project 2 : Iris Flower Classification, CodSoft Internship
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Iris%20Flower%20Classification.ipynb)
## Description :
The Iris flower dataset consists of three species: setosa, versicolor, and virginica. These species can be distinguished based on their measurements. The aim is to classify iris flowers among three species (Setosa, Versicolor, or Virginica) from the sepals' and petals' length and width measurements. 
I have used **Support Vector Machine**, **K Nearest Neighbors Classier** and **logistic Regression** for building Iris flower classification model.
## Dataset:
[https://www.kaggle.com/datasets/arshid/iris-flower-dataset]

# Project 3 : Sales Prediction Using Python, CodSoft Internship.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Sales%20Prediction%20Using%20Python%20Final.ipynb)
## Description :
Sales prediction involves forecasting the amount of a product that customers will purchase, taking into account various factors such as advertising expenditure, target audience segmentation, and advertising platform selection.
I have used **Standard Scaler** and **Sequential Artificial Neural Network** for building Sales Prediction Model.
## Dataset:
[https://www.kaggle.com/datasets/yashpaloswal/ann-car-sales-price-prediction]

# Project 4: Titanic Survival Prediction, CodSoft Internship.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Titanic%20Survival%20Prediction_Final.ipynb)
## Description :
The aim is to use the Titanic dataset and build a model that predicts whether a passenger on the Titanic survived or not. 
I have used **Random Forest Classifier(Entropy)**, **Naive Bayes Gaussian Algorithm** and **Logistic Regression** here.
## Dataset:
The dataset typically used for this project contains information about individual passengers, such as their age, gender, ticket class, fare, cabin, and whether or not they survived.
[https://www.kaggle.com/datasets/brendan45774/test-file]

# Project 5: Anomaly Detection in Transactions
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Anomaly%20Detection%20in%20Transactions.ipynb)
## Description:
Anomaly detection in transactions means identifying unusual or unexpected patterns within transactions or related activities. These patterns, known as anomalies or outliers, deviate significantly from the expected norm and could indicate irregular or fraudulent behaviour.
Here we are training an anomaly detection model using the Isolation Forest algorithm. First, we selected the relevant features for detection, namely Transaction_Amount, Average_Transaction_Amount, and Frequency_of_Transactions. We split the dataset into features (X) and the target variable (y), where X contains the selected features and y contains the binary labels indicating whether an instance is an anomaly or not. Then, we further split the data into training and testing sets using an 80-20 split ratio. Next, we created an **Isolation Forest model** with a specified contamination parameter of 0.02 (indicating the expected ratio of anomalies) and a random seed for reproducibility.
## Dataset :
[https://statso.io/anomaly-detection-case-study/]

# Project 6: Classification With Neural Network
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Classification%20with%20Neural%20Network.ipynb)
## Description :
Here, I will be using the famous MNIST fashion dataset, which contains 70,000 clothing fashion images. Here our task is to train an image classification model with neural networks.
Here, I have used **neural networks** to train an image classification model.
## Dataset
Here, I will be using the famous MNIST fashion dataset, which contains 70,000 clothing fashion images.
[https://www.kaggle.com/datasets/zalando-research/fashionmnist]

# Project 7: Language Detection with Machine Learning
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Language%20Detection%20Model.ipynb)
## Description :
Language detection is a natural language processing task where we need to identify the language of a text or document.
As this is a problem of multiclass classification, so I have used the **Multinomial Naïve Bayes** algorithm to train the language detection model as this algorithm always performs very well on the problems based on multiclass classification.
## Dataset
The dataset that I am using is collected from Kaggle, which contains data about 22 popular languages and contains 1000 sentences in each of the languages.

# Project 8: Credit Card Clustering with Machine Learning
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Credit%20Card%20Clustering%20with%20Machine%20Learning.ipynb)
## Description :
Credit card clustering is the task of grouping credit card holders based on their buying habits, credit limits, and many other financial factors. It is also known as credit card segmentation. Such clustering analysis helps businesses find their potential customers and many more marketing strategies. 
Here, I am using **K-means Clustering** for grouping of credit card customers . The groups formed range from 0 to 4.
## Dataset
The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.
[https://www.kaggle.com/datasets/arjunbhasin2013/ccdata]

# Project 9:  Weather Forecasting using Python
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Weather%20Forecasting%20.ipynb)
## Description :
weather forecasting is an application of **Time Series Forecasting** where we use time-series data and algorithms to make forecasts for a given time.With the use of weather data and algorithms, it is possible to predict weather conditions for the next n number of days.
I have used **Facebook prophet model** for this task. The Facebook prophet model is one of the best techniques for time series forecasting. 
## Dataset
This dataset provides data from 1st January 2013 to 24th April 2017 in the city of Delhi, India. The 4 parameters here are meantemp, humidity, wind_speed, meanpressure.
[https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data]

# Project 10: Cryptocurrency Price Prediction
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Cryptocurrency%20Price%20Prediction.ipynb)
## Description : 
Buying and selling result in a change in the price of any cryptocurrency, but buying and selling trends depend on many factors. Using machine learning for cryptocurrency price prediction can only work in situations where prices change due to historical prices that people see before buying and selling their cryptocurrency. 
Predicting the future prices of cryptocurrency is based on the problem of Time series analysis. The AutoTS library in Python is one of the best libraries for time series analysis. So here I will be using the AutoTS library to predict the bitcoin prices for the next 30 days
## Dataset:
For this task, I will collect the latest Bitcoin prices data from **[Yahoo Finance]**, using the **[yfinance API]**. This will help you collect the latest data each time you run this code.So the dataset contains 731 rows, where the first row contains the names of each column. 

# Project 11: Fake News Detection
## Description :
Fake news is one of the biggest problems because it leads to a lot of misinformation in a particular region. I’ll use the **Multinomial Naive Bayes** algorithm to train the fake news detection model.
## Dataset:
The dataset I am using here for the fake news detection task has data about the news title, news content, and a column known as label that shows whether the news is fake or real.
[https://www.kaggle.com/hassanamin/textdb3/download]

# Project 12: Online Payment Fraud Detection
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Online%20Payment%20Fraud%20Detection.ipynb)
## Description :
Aim was to identify online payment fraud with machine learning, we need to train a machine learning model for classifying fraudulent and non-fraudulent payments. 
I have used **Decision Tree Classifier** to classify fraud and non-fraud transactions.
## Dataset:
For this task, I collected a dataset from Kaggle, which contains historical information about fraudulent transactions which can be used to detect fraud in online payments.
[https://www.kaggle.com/ealaxi/paysim1/download]

# Project 13: Real-Time Sentiment Analysis.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Real-Time%20Sentiment%20Analysis.ipynb)
## Description :
Sentiment Analysis is an application of natural language processing that is used to understand people’s opinions. Today, many companies use real-time sentiment analysis by asking users about their service.
The main purpose of sentiment analysis is to analyze the opinions of users of a particular product or service, which helps customers understand the quality of the product. 
So, I will be using the **SentimentIntensityAnalyzer()** class provided by the NLTK library in Python.
## Dataset:
To analyze feelings in real-time, we need to request input from the user and then analyze user feelings given by him/her as input. So for this real-time sentiment analysis task using Python, I will be using the **NLTK library** in Python which is a very useful tool for all the tasks of natural language processing.

# Project 14: Retail Price Optimization Using Python.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Retail%20Price%20Optimization%20Using%20Python.ipynb)
## Description :
Retail price optimization involves determining the optimal selling price for products or services to maximize revenue and profit. 
Here, I have used **Decision Tree Regressor** Machine Learning model for the task of Retail Price Optimization.
## Dataset:
[https://statso.io/retail-price-optimization-case-study/]

# Project 15: Salary Prediction Using Python.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Salary%20Prediction.ipynb)
## Description :
For salary prediction, we need to find relationships in the data on how the salary is determined. I will train a regression model using **Linear Regression** to predict salary with Machine Learning.
## Dataset:
[https://statso.io/salary-prediction-case-study/]

# Project 16: Student Marks Prediction Model.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Student%20Marks%20Prediction%20Model.ipynb)
## Description :
Student marks prediction is a popular data science case study based on the problem of regression. I built **Linear Regression** machine learning model for prediction of student marks.
## Dataset:
The dataset I am using for the student marks prediction task is downloaded from Kaggle.
[https://www.kaggle.com/datasets/yasserh/student-marks-dataset]

# Project 17 : To Forecast hourly bike rental Demand, Internshala Trainee.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Forecast_Hourly_bike_rentalDemand.ipynb)
## Description:
Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able to rent a bike from one location and return it to a different place on an as-needed basis. Currently, there are over 500 bike-sharing programs
around the world.The data generated by these systems makes them attractive for researchers because the duration of travel, departure location, arrival location, and time elapsed is explicitly recorded. Bike sharing systems therefore function as a sensor network, which can be used for studying mobility in a city.
In this project, Our task is to combine historical usage patterns with weather data in order to forecast hourly bike rental demand.
Here, I have used **Linear Regression** and **Decision Tree** for forecasting hourly bike rental Demand.

# Project 18 : To predict if the client will subscribe to term deposit, Internshala Trainee.
(https://github.com/KhushiML/Data_ScienceProjects/blob/main/Term_Deposit_Subscription_Classifiaction.ipynb)
## Description:
Your client is a retail banking institution. Term deposits are a major source of income for a bank. A term deposit is a cash investment held at a financial institution. Your money is invested for an agreed rate of interest over a fixed amount of time, or term.
The bank has various outreach plans to sell term deposits to their customers such as email marketing, advertisements, telephonic marketing and digital marketing.Telephonic marketing campaigns still remain one of the most effective way to reach out to people. However, they require huge investment as large call centers are hired to actually execute these campaigns. Hence, it is crucial to identify the customers most likely to convert beforehand so that they can be specifically targeted via call.
Here, I have used **Logistic Regression** and **Decision Tree Algorithms**.
## Dataset :
You are provided with the client data such as : age of the client, their job type, their marital status, etc. Along with the client data, you are also provided with the information of the call such as the duration of the call, day and month of the call, etc. Given this information, our task is to predict if the client will subscribe to term deposit.

# Project 19 : Text Summarization using NLP

## Description:
Text summarization is the process of distilling the most important information from a source text.

