# Project Title

# Executive Summary: 
We are predicting the quality of red and white wines from the north of portugal, using machine learning techniques.  We will use the UCI data sets to improve our model, and our ability to predict the quality of the wines. We are analysing the data looking for a specific outcome of good or bad quality.  (Based on expert reveiws) (Move UCI data down to resource section?)

# Table of Contents

# Introduction
Any background info on the project. What is the imporrtance of wine quality prediction?

# Resources-Data set summary.  
The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].  Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

# Data Collection and Cleaning
The methods we used for our collection, steps to clean and prepprocess data. How did we hadle the missing values and outliers?

# Exploratory Data Analysis
Our initial explorations in addition to any key insights or patterns that we found.

# Model Implementation
Overview of the machine learning models used, how we implemented them. Could add code snppets her for example of the model training

# Evaluation Metrics
Explain what were using like accuracy, balanced accuracy and other metrics.

# Feature importance 
Analysis of feature importance, this is where the visualization of the top features should go

# Future Work
Potential improvements or any next steps. I think this is where adding our weather data and region can go. Any other questions we might have.

# Conclusion
This should probably match with the presentation, not sure exactly what we have found here. Maybe add something on how we think this data might impact the wine industry

# References
Data sets,tools used, any writings we might have

# Project Overview:
This project aims to predict the quality of wine based on various chemical properties using machine learning techniques. The wine quality dataset from the UCI machine learning repository is analyzed and processed. 

The goal is to build a robust predictive model to classify wine quality ratings acording to scientific standards.



# Target variable:
Our goal target variable is "quality" to determine if a wine will be rated good or bad. Thresholds were assigned by choice based on a 3 to 9 scale, we decided to use 6 to 7 to error on ratings being higher based on 5 and 6 having the highest data counts.  We wanted the mediocre wines to be qualified as "bad".

# Other Version?
# Target Variable:
Our target variable is 'quality', which aims to determine if a wine will be rated good or bad. The quality ratings in our data set range from 3 to 9. To classify the wines, we chose a threshold where wines rate 6 and above are considered 'good', while those rated below 6 are considered 'bad'. This decision was based on the distribution of the ratings where, 5 and 6 had the highest counts. By setting this threshold, we ensure that wines rated as mediocre are classified as 'bad', allowing us to focus on distinguishing the higher quality wines frome the rest.




# Requirements:
- python  
- pandas  
- sklearn  
	- metrics  
		- accuracy_score  
		- balanced_accuracy_score  
		- classification_report  
	- model_selection  
		- train_test_split  
	- utils  
		- resample  
	- preprocessing  
		- StandardScaler  
	- ensemble  
		- RandomForestClassifier  
		- LogisticRegression  
		- SVC  
		- GradientBoostingClassifier  
		- AdaBoostClassifier  
- matplotlib  
- seaborn  


License: UC Irvine Machine Learning Repository  
This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.  
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.  
DOI: 10.24432/C56S3T  
https://archive.ics.uci.edu/dataset/186/wine+quality  
archive.ics.uci.eduarchive.ics.uci.edu  
UCI Machine Learning Repository  
Discover datasets around the world!  

# Instructions 
1. Install the requirements

2. Run main.ipynb file

# Conclusion
We were sucessful in predicting to a 93.6% balanced accuracy score for our RandomForestClassifier tuned model.
Baseline and best score summary visualilaztion.





![alt text](./presentation/baseline.png)  


![alt text](./presentation/importance.png)  


![alt text](./presentation/max_depth.png)  


![alt text](./presentation/min_leaf.png)  
  
  
![alt text](./presentation/n_estimators.png)  
  
  
![alt text](./presentation/min_split.png)  
  
  
![alt text](./presentation/max_leaf.png)  
  

![alt text](./presentation/final_results.png) 

