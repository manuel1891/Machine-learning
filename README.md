# Machine-learning

## 1. Zillow Prize: Zillow’s Home Value Prediction (Zestimate) (Part 1)

The first part or this machine learning competition consists of building a model to improve the Zestimate residual error.

*Zestimate* is the estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property.

With a script that I made in R, I finished in the best 30% of the competition and am now translating this script in Python. This is work in progress.


## 2. BNP - Kaggle Challenge - Can we accelerate BNP Paribas Cardif's claims management process?
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
