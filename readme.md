Credit Loan Deafault Risk Prediction
====

Business Goal
===
My project is about the prediction of credit loan deafault, which could be used by financial service companys to make business decision about whether they should accept given loan application or not.<br>

Data Ingestion
===
The dataset is downloaded from LendingClub website (https://www.lendingclub.com/info/download-data.action). The dataset I used is 2018 Q1 loan status dataset, which is 123.4 Mb large and contains 107866 records and 150 features. The data preprocessing of my project includes 4 main parts: filling null values, standardizing numerical variables, one hot encoding for all categorical variables and removing outliers.

Visualizations
===
At EDA stage, I generate several graphs and tables to better understand the dataset.

Machine Learning:
===
Before actual modeling, since the target variable is highly unbalance distributed, the naive model (simply predicting all loans would not default) has a high accuracy at about 90%, which could be used as a benchmark for all models I made. The models I used include Logistic Regression (class weight balanced), Gradient Boosting Trees (oversampling), Random Forest and neural network. After parameter tuning process, I build an ensemble model based on the weighted average of above four models combined, and the accuracy of which finally reaches 97.7%, with precision at 89.6% and recall at 87.1%.

Repo structure
===
The jupyter notebook includes all visualization and its easy to read. The py file could be used to generate all models in models_save folder (the RF model is too big to upload to github). The data folder contains original data file in csv format.
