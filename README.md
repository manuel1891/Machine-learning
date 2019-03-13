# Machine-learning

## 1. Zillow Prize: Zillow’s Home Value Prediction (Zestimate) (Part 1)

The first part or this machine learning competition consists of building a model to improve the Zestimate residual error.

*Zestimate* is the estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property.

With a script that I made in R, I finished in the best 30% of the competition and am now translating this script in Python. This is work in progress.



## 2. Santander - Kaggle Challenge - Products recommendation engine
(Kubrick Group)

*Notice that this project is based on a publicly available script by author SRK.*

By groups of 2, we had to build a recommendation engine that recommends products  to Santander customers, in order to increase 
revenues, traffic and customer satisfaction. 

We tweaked the original model and parameters, and added up to 5 lags of the diverse products, which were significant predictors.
We used XGBoost to run the model and finished in the top 10% of the competition. 


## 3. BNP - Kaggle Challenge - Can we accelerate BNP Paribas Cardif's claims management process?
(The University of St Andrews)

By groups of 5, we had to build an engine that automatically classifies BNP claims into two categories: 

                            1 - can be approved for faster payment
  
                            0 - requires more investigation before approval

The aim of this project is to:    * Improve customer experience
                                  * Reduce costs
                                  * Smarten existing systems

  
We investigated a few machine learning algorithms:
  * Decision Tree
  * Bagging
  * Random Forest
  * Boosting 
  * Neural Nets
  
The Random Forest model (500 iterations) yielded the best result (84.5 % accuracy on the training set and 47.3% logloss on the 
test set). The model returns probabilities for the target variable. Accepting the claims having probability of 70% and higher 
for faster payment means that 66% of the claims receive approval for faster payment. 


## 2. Santander - Kaggle Challenge - Products recommendation engine
(Kubrick Group)

*Notice that this project is based on a publicly available script by author SRK.*

By groups of 2, we had to build a recommendation engine that recommends products  to Santander customers, in order to increase 
revenues, traffic and customer satisfaction. 

We tweaked the original model and parameters, and added up to 5 lags of the diverse products, which were significant predictors.
We used XGBoost to run the model and finished in the top 10% of the competition. 



