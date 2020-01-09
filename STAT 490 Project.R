#Necessary libraries
library(pander)
library(tidyr)
library(broom)
library(lmSupport)
library(caret)#
library(leaps)
library(elasticnet)
library(rpart)
library(rpart.plot)
library(randomForest)
library(class)
library(car)

house <- read.csv("kc_house_data.csv", header = T)
house2 <- read.csv("kc_house_data.csv", header = T) #Extra dataset
dim(house)
set.seed(2018)

#Creating scatterplots for each predictor to visualize relationships
for(i in 1:20){
  housefit <- lm(price ~ house[,i], data = house)
  plot(price ~ house[,i], data = house)
  abline(housefit, col = 'red')
}

#Adjusting the dataset
house$id <- NULL
house <- house[-c(15871),] #Typo for bathrooms at this data point
house$yr_built <- as.numeric(substr(house$date, start = 1, stop = 4)) - house$yr_built
house$yr_renovated <- ifelse(house$yr_renovated == 0,
                             house$yr_built,
                             as.numeric(substr(house$date, start = 1, stop = 4)) - house$yr_renovated)
house$bed_bath_ratio <- ifelse(house$bathrooms == 0,
                               house$bedrooms,
                               house$bedrooms / house$bathrooms)
house$sqft_lot <- 1/(house$sqft_lot) #1/X transformation on sqft_lot
house$sqft_lot15 <- 1/(house$sqft_lot15) #1/X transformation on sqt_lot15
  #Changing lat and long to dummy variables
house$long <- ifelse(house$long <= -122.1, 1, 0)
house$lat <- ifelse(house$lat >= 47.5, 1, 0)

summary(house)

#NOTE: Typo for bedroom at row 15872

#Adjusting extra dataset
house2 <- house[-c(15871),] #Typo for bathrooms at this data point
house2$yr_built <- as.numeric(substr(house2$date, start = 1, stop = 4)) - house2$yr_built
house2$yr_renovated <- ifelse(house2$yr_renovated == 0,
                             house2$yr_built,
                             as.numeric(substr(house2$date, start = 1, stop = 4)) - house2$yr_renovated)
house2$bed_bath_ratio <- ifelse(house2$bathrooms == 0,
                               house2$bedrooms,
                               house2$bedrooms / house2$bathrooms)
house2$sqft_lot <- 1/(house2$sqft_lot) #1/X transformation on sqft_lot
house2$sqft_lot15 <- 1/(house2$sqft_lot15) #1/X transformation on sqt_lot15
  #Changing lat and long to dummy variables
house2$long <- ifelse(house2$long <= -122.1, 1, 0)
house2$lat <- ifelse(house2$lat >= 47.5, 1, 0)

house2$date <- NULL
house2$id <- NULL
house2$zipcode <- NULL

#Only run these 2 after running feature selection 
house2.train$sqft_above <- NULL
house2.train$sqft_basement <- NULL

#Only run these 3 after running regression trees on first adjusted model
house2.train$floors <- NULL
house2.train$sqft_lot15 <- NULL
house2.train$yr_renovated <- NULL

#Creating training and test data
index <- sample(nrow(house), size = 0.7*nrow(house))
house.train <- house[index, ] #training data
dim(house.train)
house.test <- house[-index, ] #test data
dim(house.test)

index2 <- sample(nrow(house2), size = 0.7*nrow(house2))
house2.train <- house2[index2, ]
house2.test <- house[-index2, ]

#Initial fit with all predictors

house.fitall <- lm(price ~ bedrooms + bathrooms + bed_bath_ratio + sqft_living +
                   sqft_lot + floors + waterfront + view + condition + grade +
                   yr_built + yr_renovated + lat + sqft_above + sqft_basement + 
                   long + sqft_living15 + sqft_lot15, data = house.train)
summary(house.fitall)

#Assessing multicollinearity
vif(house.fitall)
alias(house.fitall)
#Figured out sqft_living and sqft_above are linearly dependent

#Assessing multicollinearity between sqft_living and sqft_above
house.fit.sqlivingabove <- lm(sqft_living ~ sqft_above, data = house)
summary(house.fit.sqlivingabove)
plot(sqft_above ~ sqft_living, data = house, xlab = "sqftliving", ylab = "sqftabove")
abline(house.fit.sqlivingabove, col = 'red')

#Both methods run first with predictors 'sqft_above' 
#and 'sqft_basement'
#Second time run without these variables.
#Third time run without predictors yr_renovated, floors, and sqft_lot15
#Ridge Regression
house.trainControl <- trainControl(method = "cv", number = 5)
ridge.house <- train(price ~.,
                     data = house2.train,
                     method = 'ridge',
                     tuneLength = 20,
                     trControl = house.trainControl,
                     preProcess = c('center', 'scale'))

ridge.house
ridge.house$bestTune
ridge.house$finalModel$beta.pure
plot.enet(ridge.house$finalModel,
          xvar="penalty", use.color = TRUE)

#LASSO
house.trainControl.lasso <- trainControl(method = 'cv', number = 5)
lasso.house <- train(price ~ .,
                     data = house2.train,
                     method = 'lasso',
                     tuneLength = 20,
                     trControl = house.trainControl.lasso,
                     preProcess = c('center', 'scale'))
lasso.house
lasso.house$bestTune
lasso.house$finalModel$beta.pure
plot.enet(lasso.house$finalModel,
          xvar = "penalty", use.color = TRUE)

#Based on linear dependency between sqft_living and sqft_above,
#sqft_above shown to be less significant through ridge regression
#and LASSO ,so this and 
#sqft_basement were removed from the initial model, as 
#sqft_living accounts for both of these

#Feature selections first run with sqft_above and sqft_basement
#then run without them
#Best subset selection
house.regbs.out <- 
  regsubsets(price ~ bedrooms + bathrooms + bed_bath_ratio + sqft_living +
               sqft_lot + floors + waterfront + view + condition + grade +
               yr_built + yr_renovated + lat +
               long + sqft_living15 + sqft_lot15, 
             data = house.train,
             nbest = 1, nvmax = NULL,
             method = "exhaustive")
house.regbs.sum <- summary(house.regbs.out)
data.frame(p = 1:16, adj.R2 = house.regbs.sum$adjr2,
           Cp = house.regbs.sum$cp, BIC = house.regbs.sum$bic)
which.max(house.regbs.sum$adjr2)
which.min(house.regbs.sum$cp)
which.min(house.regbs.sum$bic)
data.frame(adj.R2 = house.regbs.sum$which[14,],
           Cp = house.regbs.sum$which[14,],
           BIC = house.regbs.sum$which[14,])

house.regbs.sum$outmat

#Forward selection
house.regforw.out <- 
  regsubsets(price ~ bedrooms + bathrooms + bed_bath_ratio + sqft_living +
               sqft_lot + floors + waterfront + view + condition + grade +
               yr_built + yr_renovated + lat + 
               long + sqft_living15 + sqft_lot15, 
             data = house.train,
             nbest = 1, nvmax = NULL,
             method = "forward")
house.regforw.sum <- summary(house.regforw.out)
data.frame(p = 1:16, adj.R2 = house.regforw.sum$adjr2,
           Cp = house.regforw.sum$cp, BIC = house.regforw.sum$bic)
which.max(house.regforw.sum$adjr2)
which.min(house.regforw.sum$cp)
which.min(house.regforw.sum$bic)
data.frame(adj.R2 = house.regforw.sum$which[14,],
           Cp = house.regforw.sum$which[14,],
           BIC = house.regforw.sum$which[14,])

house.regforw.sum$outmat

#Backward selection
house.regback.out <- 
  regsubsets(price ~ bedrooms + bathrooms + bed_bath_ratio + sqft_living +
               sqft_lot + floors + waterfront + view + condition + grade +
               yr_built + yr_renovated + lat +
               long + sqft_living15 + sqft_lot15, 
             data = house.train,
             nbest = 1, nvmax = NULL,
             method = "backward")
house.regback.sum <- summary(house.regback.out)
data.frame(p = 1:16, adj.R2 = house.regback.sum$adjr2,
           Cp = house.regback.sum$cp, BIC = house.regback.sum$bic)
which.max(house.regback.sum$adjr2)
which.min(house.regback.sum$cp)
which.min(house.regback.sum$bic)
data.frame(adj.R2 = house.regback.sum$which[14,],
           Cp = house.regback.sum$which[14,],
           BIC = house.regback.sum$which[14,])

house.regback.sum$outmat

data.frame(best.subset = house.regbs.sum$which[14,],
           forward = house.regforw.sum$which[14,],
           backward = house.regforw.sum$which[14,])

#After running ridge/LASSO/feature selection w/out sqft_above and 
#sqft_basement, decided predictors floors, sqft_lot15, and yr_renovated
#are out. First two because of feature selection (only two out 
#because of BIC, last one because it was the first one eliminated
#in ridge/LASSO methods, and would have been next predictor out
#with feature selection.

#Model after running ridge regression/ LASSO/ feature selection
house.fitadj <- lm(price ~ bedrooms + bathrooms + bed_bath_ratio +
     sqft_living + sqft_lot + waterfront + view + condition + grade +
     yr_built + lat + long + sqft_living15, data = house.train)
summary(house.fitadj)
vif(house.fitadj)
plot(house.fitadj)
plot(house.fitadj, plot = "5")

#Negligible test MSE increase - adds validity to model

#Regression Trees/ Tree Pruning

house$sqft_above <- NULL
house$sqft_basement <- NULL
house$floors <- NULL
house$sqft_lot15 <- NULL
house$yr_renovated <- NULL
house.train$date <- NULL
house.train$zipcode <- NULL
house.train$sqft_above <- NULL
house.train$sqft_basement <- NULL
house.train$floors <- NULL
house.train$sqft_lot15 <- NULL
house.train$yr_renovated <- NULL

house.fit.tree <- rpart(price ~ ., data = house.train)
house.fit.tree
prp(house.fit.tree)

#Prediction on the training and testing data
yhat.train <- predict(house.fit.tree)
yhat.test <- predict(house.fit.tree, newdata = house.test)
#Test MSE
(test.rmse <- sqrt(mean((house.test$price - yhat.test)^2)))
#test.mse <- mean((Credit[-train,]$Balance - yhat.test)^2)

#Regression Tree Performance on 5-fold CV Data Partition
#Creating indices for 5-fold CV
house.flds <- caret::createFolds(1:nrow(house), k = 5,
                           list = TRUE, returnTrain = TRUE)
house.cv.rmse <- rep(NA,5)
for(i in 1:5){
  #use training indices generated above
  house.fit.cv <- rpart(price~., data = house, subset = house.flds[[i]])
  #predict responses in test data
  yhat.t <- predict(house.fit.cv, newdata = house[-house.flds[[i]],])
  house.cv.rmse[i] <- sqrt(mean((house[-house.flds[[i]],]$price - yhat.t)^2))
}

summary(house.cv.mse)
printcp(house.fit.tree)
#CV training MSE
1.3359e+11 * .32345

#CV test MSE
1.3359e+11 * .35364

#Display just error tables
house.fit.tree$cptable

#check which tree size satisfies 'xerror < lowxerror + lowxstd'
which.min(house.fit.tree$cptable[,4])
house.fit.tree$cptable[,4] < house.fit.tree$cptable[10,4] + house.fit.tree$cptable[10,5]

#Pruning
data.frame(CP = house.fit.tree$cptable[,1],
           xerror = house.fit.tree$cptable[,4],
           x.error.std = house.fit.tree$cptable[,4] + house.fit.tree$cptable[,5])

#Estimated cv error
214736 * .1725058

#Plot of CV training error
plotcp(house.fit.tree, cex.lab = 0.75, cex.axis = 0.75)
#Pruned tree, cp is complex parameter
(house.fit.tree1 <- prune(house.fit.tree, cp = 0.014))
#To ensure we get tree 10, the CP value we give to the
#pruning algorithm is the geometric midpoint of CP values
#for tree 10 and tree 9
(cp.opt <- sqrt(house.fit.tree$cptable[8,1] * house.fit.tree$cptable[9,1]))
(house.fit.tree2 <- prune(house.fit.tree, cp = cp.opt))

#Plot trees
par(mfcol = c(1,3))
prp(house.fit.tree)
prp(house.fit.tree1)
prp(house.fit.tree2)

#Prediction on testing data
yh0 <- predict(house.fit.tree, house.test)
yh1 <- predict(house.fit.tree1, house.test)
yh2 <- predict(house.fit.tree2, house.test)
#Test RMSE
data.frame(
  cp0.01 = sqrt(mean((house.test$price - yh0)^2)),
  cp0.02 = sqrt(mean((house.test$price - yh1)^2)),
  cp.opt = sqrt(mean((house.test$price - yh2)^2))
)

#Fit tree models on 5-fold CV test data sets
#Creating indices for 5-fold CV
house.flds2 <- caret::createFolds(1:nrow(house), k = 5,
                            list = TRUE, returnTrain = TRUE)
house.varp <- function(x) mean((x-mean(X))^2) #Compute total error
house.rmse.cv <- matrix(NA, nrow = 4, ncol = 5)
for(i in 1:5){
  tr.cv <- house.flds2[[i]]
  fit.cv2 <- rpart(price~., data = house, subset = tr.cv)
  yh0 <- predict(fit.cv2, house[-tr.cv,])
  yh1 <- predict(prune(fit.cv2, cp = 0.013), house[-tr.cv,])
  yh2 <- predict(prune(fit.cv2, cp = cp.opt), house[-tr.cv,])
  yhat.t <- predict(fit.cv2, newdata = house[-flds[[i]],])
  house.rmse.cv[,i] <- c(
    fit.cv2$cptable[5,"xerror"] * house.varp(house[tr.cv,]$price),
    sqrt(mean((house[-tr.cv,]$price - yh0)^2)),
    sqrt(mean((house[-tr.cv,]$price - yh1)^2)),
    sqrt(mean((house[-tr.cv,]$price - yh2)^2)))
}
rownames(house.rmse.cv) <- c("house.cv.mse", "tmse-cp0.01",
                      "tmse-cp0.02", "tmse-cp.opt")
house.rmse.cv
apply(house.rmse.cv, 1, mean)

#Random Forests
  #Get trees in forest
#(p1 <- getTree(house.fit.rf, k = 1, labelVar = TRUE))

house.fit.rf2 <- randomForest(price~., data = house.train,
                             mtry = 2,
                             ntree = 150,
                             importance = TRUE)
house.fit.rf3 <- randomForest(price~., data = house.train,
                              mtry = 3,
                              ntree = 150,
                              importance = TRUE)
house.fit.rf4 <- randomForest(price~., data = house.train,
                              mtry = 4,
                              ntree = 150,
                              importance = TRUE)
house.fit.rf5 <- randomForest(price~., data = house.train,
                              mtry = 5,
                              ntree = 150,
                              importance = TRUE)
  #Out of Bag Test MSE
par(mfcol = c(1,1))
plot(house.fit.rf3, main = "Out of Bag Test MSE")

  #Print Results
print(house.fit.rf2)
importance(house.fit.rf2)
varImpPlot(house.fit.rf2, main = "")

print(house.fit.rf3)
importance(house.fit.rf3)
varImpPlot(house.fit.rf3, main = "")

print(house.fit.rf4)
importance(house.fit.rf4)
varImpPlot(house.fit.rf4, main = "")

print(house.fit.rf5)
importance(house.fit.rf5)
varImpPlot(house.fit.rf5, main = "")

yh.rf2 <- predict(house.fit.rf2, newdata = house.test)
yh.rf3 <- predict(house.fit.rf3, newdata = house.test)
yh.rf4 <- predict(house.fit.rf4, newdata = house.test)
yh.rf5 <- predict(house.fit.rf5, newdata = house.test)

  #Test RMSE's for random forests
data.frame(
  rf.mse2 = sqrt(mean((house.test$price - yh.rf2)^2)),
  rf.mse3 = sqrt(mean((house.test$price - yh.rf3)^2)),
  rf.mse4 = sqrt(mean((house.test$price - yh.rf4)^2)),
  rf.mse5 = sqrt(mean((house.test$price - yh.rf5)^2))
)

  #Test RMSE tanks like a rock with regression trees
  #Importance difference between bathrooms, bedrooms, bed_bath_ratio
  #negligible - not removing any of these based off of this

#Final model
house.fitfinal <- lm(price ~ bed_bath_ratio + sqft_living +
                     waterfront + view + condition + grade +
                     yr_built + lat + long + sqft_living15, data = house.train)
summary(house.fitfinal)
vif(house.fitfinal)

#Calculating test MSE for each model
house.predfitall <- predict(house.fitall, newdata = house.test)
house.all.rmse <- sqrt(mean((house.test$price - house.predfitall)^2))

house.predfitadj <- predict(house.fitadj, newdata = house.test)
house.adj.rmse <- sqrt(mean((house.test$price - house.predfitadj)^2))

house.predfitfinal <- predict(house.fitfinal, newdata = house.test)
house.final.rmse <- sqrt(mean((house.test$price - house.predfitfinal)^2))

house.all.rmse
house.adj.rmse
house.final.rmse
