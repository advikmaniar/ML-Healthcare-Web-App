# ML-Healthcare-Web-App

This is an interactive Machine Learning Web App "ML in Healthcare" developed using Python and StreamLit. It uses ML algorithms to build powerful and accurate models to predict the risk (High / Low) of the user of having a Heart Attack or Breast Cancer based on the user's specific attributes like age, sex, heart rate, blood sugar, etc.

<h2><b> View App Here: </b></h2>

[![StreamLit App](https://static.streamlit.io/badges/streamlit_badge_white.svg)](https://share.streamlit.io/advikmaniar/ml-healthcare-web-app/main/ML_Healthcare.py)

<hr>

This applications has two basic sections:

<h2>1) - Model Building </h2>
In this section 7 different models are built using different ML algorithms. They are: 

```
1. Logistic Regression 
2. KNN
3. SVM 
4. Decision Trees 
5. Random Forest 
6. Gradient Boosting 
7. XGBoost
```
The models are trained using data from https://archive.ics.uci.edu/ml/index.php, particularly the [Heart Attack Prediction](https://github.com/advikmaniar/ML-Heathcare-Web-App/blob/main/Data/heart.csv) and [Breast Cancer (Wisconsin)](https://github.com/advikmaniar/ML-Heathcare-Web-App/blob/main/Data/BreastCancer.csv) datasets.

An interactive side-dashboard is created using the streamlit `st.sidebar` call which enables the user to do the following:
1. Choose dataset - `Heart Attack / Breast Cancer`
2. Choose algorithm - `Logistic Regression , KNN , SVM , Decision Trees , Random Forest , Gradient Boosting , XGBoost.`
3. Change the important parameters for each model - `Learning Rate, Random State, Regularization Coeff, Gamma, Kernel, n_estimators` etc. 

After training using the parameters selected by the user, the tuned model is built and ready to be tested on our testing data. The classification plot and confusion matrix is displayed for the model selected along with the model metrics: `Accuracy, Precision, Recall, F1-Score, Mean Squared Error, Execution Time`. The user can observe real-time changes in the plots and metrics as they change the model parameters further. 
> **This is a great way to understand the different ML algorithms and how they are influenced by tuning the hyperparameters.**
> 
![image](https://user-images.githubusercontent.com/72503778/123002403-85b73700-d3cf-11eb-80a1-71262561b9c8.png)

The 7 models (optimum tuning) performed as follows: <br>
`Criterion: Accuracy`
Model | Accuracy (Heart Attack / Breast Cancer)
------------ | -------------
Logistic Regression | **91.803% / 100.0%**
KNN | **86.89% / 96.49%**
SVM | **93.44% / 100.0%**
Decision Trees | **52.56% / 60.53%**
Random Forest | **90.164% / 98.24%**
Gradient Boosting | **88.53% / 96.49%**
XGBoost | **95.08% / 94.737%**

<h2>2) - User Prediction </h2>
In this section, the user can use any model developed above to predict their status (High Risk / Low Risk) using their own values. (Either for Heart Attack or Breast Cancer)

![image](https://user-images.githubusercontent.com/72503778/123003157-6d93e780-d3d0-11eb-81fc-8dd6abe89efa.png)

![image](https://user-images.githubusercontent.com/72503778/123003260-93b98780-d3d0-11eb-9ff0-bb27da6a105e.png)


View the final video [here](https://github.com/advikmaniar/ML-Healthcare-Web-App/blob/main/Results/Video.mp4).
<hr>

<h1> Thank You! </h1>

<hr>



