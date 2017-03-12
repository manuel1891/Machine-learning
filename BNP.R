###########################################  GROUP GREYJOY #####################################################
#Importing the data sets into R

train = read.table("train.csv", sep=",", header=T, fill=T)
test = read.table("test.csv", sep=",", header=T, fill=T)


#Identifying missing patterns in the data (ENTER CODE VIM)

#Gives the indices of the columns in train that are factors and vectors
numericIndices = which(sapply(as.list(train), is, "numeric"))
factorIndices = which(sapply(as.list(train), is, "factor"))

#Creates subdata frames with entries of a single type. "3:" cuts out ID and target.
numericTrain = train[,numericIndices[3:length(numericIndices)]]
factorTrain = train[,factorIndices]

#vifstep performs a dimensionality reduction
vifReduction = vifstep(numericTrain)
#Creates reduced dataset by excluding dependent columns.
reducedNumericTrain = exclude(numericTrain, vifReduction)

#Multivariate Imputation by Chained Equations
install.packages('mice')
library(mice)

train = reducedNumericTrain
fullTest = test
#Selects only the numeric columns and only those that we are going to impute. So not ID and target.
test = fullTest[colnames(train[4:ncol(train)])]

#Imputation
imputedData = mice(rbind(train[4:ncol(train)],test), m=2, maxit=2, meth="pmm", seed=456789)
impu=complete(imputedData,1)


#We now separate this into two files
#Train
write.csv(cbind(train[1:3],impu[1:nrow(train),]), "imputedTrain.csv")

#Test
write.csv(cbind(fullTest[1],impu[(nrow(train)+1):nrow(impu),]), "imputedTest.csv")



############# CLASSIFICATION TREES ##########################################################################
#Packages required for the classification tree
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)


#Library for classification tree
library(rpart)



#Fitting the classification tree
fit <- rpart(target ~ ., data=train, method="class")

#Plotting the classification tree
fancyRpartPlot(fit)


#Fitting the classification tree with some control features

fit2 <- rpart(target ~ ., data=train, method="class",
             control=rpart.control(minsplit=50, maxcompete = 4,maxsurrogate = 5,usesurrogate = 2))

#Cross validation for decision tree

cross_validation = function(data,k){
  folds <- cut(seq(1,nrow(data)),breaks=k,labels=FALSE)
  sum_rss=0
  #Perform 10 fold cross validation
  for(i in 1:k){
    #Segement your data by fold using the which() function 
    
    testIndexes <- which(folds==i,arr.ind=TRUE)
    
    
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    
    set.seed(111)
    
    attach(trainData)
    
    model <- rpart(target ~ ., data=trainData, method="class",
                   control=rpart.control(minsplit=50, maxcompete = 4,maxsurrogate = 5,usesurrogate = 2))
    
    detach(trainData)
    attach(testData)
    
    myPrediction <- predict(model, testData,type="prob")[,2]
    
    
    rss=0
    for(i in 1 : nrow(testData)){
      temp = myPrediction[i]-testData$target[i]
      rss = rss + temp*temp
    }
    sum_rss = rss + sum_rss
    
    detach(testData)
    print("go")
  }
  sum_rss = sum_rss/k/(nrow(data)/k)
  print("End cross_validation ")
  return(sum_rss)
}

##Perform 10 Folds cross Validation on the decision tree##
cross_validation(train,10)

#Predictions using fit

Prediction <- predict(fit, test, type = "prob")
submit <- data.frame(ID = test$ID, target = Prediction)
write.csv(submit, file = "model1.csv", row.names = FALSE)


#Predictions using fit2

Prediction <- predict(fit2, test, type = "prob")
submit <- data.frame(ID = test$ID, target = Prediction)
write.csv(submit, file = "model2.csv", row.names = FALSE)


############### RANDOM FOREST #################################################################################

#Package required for the Random Forest model
install.packages('randomForest')
library("randomForest")

#To be able to replicate the output
set.seed(111)

#Fitting the random forest with 500 trees
fit3 <- randomForest(as.factor(target)~.,data=train, importance=TRUE, ntree=500, na.action=na.fail)

##CV Function for random forest##
cross_validation = function(data,k){
  folds <- cut(seq(1,nrow(data)),breaks=k,labels=FALSE)
  sum_rss=0
  #Perform 10 fold cross validation
  for(i in 1:k){
    #Segement your data by fold using the which() function 
    
    testIndexes <- which(folds==i,arr.ind=TRUE)
    
    
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    
    set.seed(111)
    
    attach(trainData)
    
    model <- randomForest(as.factor(target)~.,data=trainData, importance=TRUE, ntree=500, na.action=na.fail)
    
    detach(trainData)
    attach(testData)
    
    myPrediction <- predict(model, testData,type="prob")[,2]
    
    
    rss=0
    for(i in 1 : nrow(testData)){
      temp = myPrediction[i]-testData$target[i]
      rss = rss + temp*temp
    }
    sum_rss = rss + sum_rss
    
    detach(testData)
    print("go")
  }
  sum_rss = sum_rss/k/(nrow(data)/k)
  print("End cross_validation ")
  return(sum_rss)
}



#Predictions using fit3
my_prediction <- predict(fit3, train ,type="prob")[,2]
my_solution <- data.frame(ID=test$ID, PredictedProb=my_prediction)
write.csv(my_solution, file="model3.csv",row.names=FALSE)


############ NEURAL NETWORK ###################################################################################

library(caret)

train = read.table("train.csv", sep=",", header=T, fill=T)
numericTrain = read.table("imputedTrain.csv", sep = ",", header = T, fill=T)
test = read.table("test.csv", sep=",", header=T, fill=T)
numericTest = read.table("imputedTest.csv", sep = ",", header = T, fill=T)

attach(train)

factorIndices = which(sapply(as.list(train), is, "factor"))

#Creates subdata frames with entries of a single type. "3:" cuts out ID and target.
factorTrain = train[,factorIndices]
factorTest = test[colnames(factorTrain)]

#missingNumericTrain = apply(numericTrain, c(1,2), is.finite)
#missingNumericTrain[1:ncol(missingNumericTrain)] = as.factor(missingNumericTrain[1:ncol(missingNumericTrain)])

#missingFactorTrain = apply(factorTrain, c(1,2), function(stringI){nchar(stringI) < 1})
#missingFactorTrain[1:ncol(missingFactorTrain)] = as.factor(missingFactorTrain[1:ncol(missingFactorTrain)])

strip = function(x)
{
  u = factor(x, rownames(sort(table(x),decreasing=TRUE))[1:5])
  levels(u) = c(levels(u), "Nathan Price is cool")
  u[is.na(u)] = "Nathan Price is cool"
  return(u)
}

binded = rbind(factorTrain, factorTest)

simpleBinded = sapply(binded, strip)
simp = data.frame(simpleBinded[,1])
for (i in 2:ncol(simpleBinded))
{
  simp = cbind(simp,factor(simpleBinded[,i]))
}
simpleBinded = simp
colnames(simpleBinded) = colnames(factorTrain)
simpleFactorTrain = simpleBinded[1:nrow(factorTrain),]
simpleFactorTest = simpleBinded[(1+nrow(factorTrain)):nrow(simpleBinded),]
#colnames(missingNumericTrain) = colnames(numericTrain)
#colnames(missingFactorTrain) = colnames(missingFactorTrain)

target = as.factor(target)

fullTrain = cbind(target, numericTrain[5:ncol(numericTrain)], simpleFactorTrain)

fit4 = train(form = target~., data = fullTrain, method='nnet', maxit=1000, trControl=trainControl(method='cv'))


#simpleFactorTest = data.frame(apply(factorTest, 2, strip))

fullTest = cbind(numericTest[3:ncol(numericTest)], simpleFactorTest)


#Predictions using fit 4
predictions = predict(fit4, fullTest, type="prob")
submission = data.frame(test$ID, predictions[2])
colnames(submission) = c("ID","PredictedProb")
write.csv(submission, "fit4.csv", row.names = FALSE)




############  Bagging #######################################################################################
install.packages("ipred")
library(ipred)

bagg<-bagging(target~., data=train, coob=T)
pred2<-predict(bagg, test, type=response)

# Cross-validation
cv_score<-cv(target, target~., train, bagging, pred2)

# Submission
submission <- data.frame(ID=test$ID, PredictedProb=pred2)
colnames(submission) = c("ID","PredictedProb")
write.csv(submission, "nondenon.csv", row.names = FALSE)



############ XGBOOST  #########################################################################################

install.packages("xgboost")
library(xgboost)
install.packages("readr")
library(readr)
install.packages("stringr")
library(stringr)
install.packages("caret")
library(caret)
install.packages("car")
library(car)


set.seed(100)

md <- 20
ss <- 0.96
cs <- 0.45
mc <- 1
np <- 1
cat("Sample data for early stopping\n")
h <- sample(nrow(train),50000)
?sample
feature.names <- names(train)[c(2:ncol(train))]
feature.names

# Replacing Categorical variables with integers 
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}
tra<-train[,feature.names]

# Transforming the data matrix
dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$target[h])
dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$target[-h])

watchlist<-list(val=dval,train=dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "logloss",
                eta                 = 0.01,
                max_depth           = md,
                subsample           = ss,
                colsample_bytree    = cs,
                min_child_weight    = mc,
                num_parallel_tree   = np
)


nrounds<-1000

# Training model
clf <- xgb.train(   data                = dtrain, 
                    nrounds             = 1000, 
                    verbose             = 1,  #1
                    early.stop.round    = 300,
                    watchlist            = watchlist,
                    maximize            = FALSE,
                    objective           = "binary:logistic")



pred1 <- predict(clf,data.matrix(test), ntreelimit=clf$bestInd)

# Submission
submission <- data.frame(ID=test$ID, PredictedProb=pred1)
colnames(submission) = c("ID","PredictedProb")
write.csv(submission, "xgboost.csv", row.names = FALSE)

# Cross-Validation
xgb.cv(params = param, data=dtrain, nrounds, nfold= 10)