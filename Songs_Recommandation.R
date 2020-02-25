library(softImpute)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)
library(reshape2)
library(rpart)
library(ggplot2)
library(caTools) 
library(rpart) 
library(rpart.plot) 
library(GGally)
library(ROCR)
library(MASS)
library(caret)
library(gbm)
#####Question a ###########

MusicRatings <- read.csv("MusicRatings.csv")
Songs <- read.csv("Songs.csv")
Users <- read.csv("Users.csv")

length(Songs$songID)
length(Users$userID)
RangeOfRatings = max(MusicRatings$rating) - min(MusicRatings$rating)
RangeOfRatings

set.seed(345)

## We create the different set (training, validation A, validation B and testing sets)
train.ids <- sample(nrow(MusicRatings), 0.92*nrow(MusicRatings))
train <- MusicRatings[train.ids,]
test <- MusicRatings[-train.ids,]

valA.ids <- sample(nrow(train), (4/92)*nrow(train))
valA <- train[valA.ids,]
train <- train[-valA.ids,]

valB.ids <- sample(nrow(train), (4/88)*nrow(train))
valB <- train[valB.ids,]
train <- train[-valB.ids,]

## We create an incomplete training set matrix 

matrix.train <- Incomplete(train$userID, train$songID, train$rating)
head(summary(matrix.train,5))

######Question b#########

#number of parameters
length(Songs$songID)+length(Users$userID)

#number of observations to train the model
length(train$rating)

#we use the function biScale to fit the model
matrix.train.standardized <- biScale(matrix.train, maxit = 1000, row.scale = FALSE, col.scale = FALSE)
head(summary(matrix.train.standardized,5))

#we remove the bias os users for rating songs highly or lowly 

alpha <- attr(matrix.train.standardized, "biScale:row")$center
beta <- attr(matrix.train.standardized, "biScale:column")$center 

Users$alpha <- alpha
Songs$beta <- beta

head(Users)
head(Songs)

#we remove alpha from the model to get the 3 highest value of Beta corresponding to the three most popular songs
Highest_betas <- order(Songs$beta, decreasing = TRUE)[1:3]

Most_popular_songs <- data.frame(ID = Songs$songID[Highest_betas], Name = Songs$songName[Highest_betas], Artist = Songs$artist[Highest_betas])
Most_popular_songs

#now we remove beta to get the 3 most enthused users
Highest_alphas <- order(Users$alpha, decreasing = TRUE)[1:3]

Most_enthused_users <- data.frame(ID = Users$userID[Highest_alphas])
Most_enthused_users

#we compute metrics to assess the performance of our model 1

predictions = alpha[test$userID] + beta[test$songID]
MAE =  mean(abs(test$rating - predictions))/RangeOfRatings
RMSE= sqrt(mean((test$rating - predictions)^2))/RangeOfRatings

OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}
OSR2_model1 = OSR2(predictions, train$rating, test$rating)

#######Question c ##########

#We determine the value of k, the number of archetypes that should be selected

mae.vals = rep(NA, 20)
for (rnk in seq_len(20)) {
  print(str_c("Trying rank.max = ", rnk))
  mod <- softImpute(matrix.train.standardized, rank.max = rnk, lambda = 0, maxit = 1000)
  preds <- impute(mod, valA$userID, valA$songID) %>% pmin(5) %>% pmax(1)
  mae.vals[rnk] <- mean(abs(preds - valA$rating))
}

mae.val.df <- data.frame(rnk = seq_len(20), mae = mae.vals)
ggplot(mae.val.df, aes(x = rnk, y = mae)) + geom_point(size = 3) + 
  ylab("Validation MAE") + xlab("Number of Archetypal Users") + 
  theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

which.min(mae.vals) #we take k=4 because it leads to the lowest value of MAE

#We build a final filtering model 

mod_2 <- softImpute(matrix.train.standardized, rank.max = 4, lambda = 0, maxit = 1000)
preds <- impute(mod_2, test$userID, test$songID) %>% pmin(5) %>% pmax(1)

mean(abs(preds - test$rating))/4
sqrt(mean((preds - test$rating)^2))/4
OSR2(preds, train$rating, test$rating)
MAE =  mean(abs(test$rating - preds))/RangeOfRatings
RMSE= sqrt(mean((test$rating - preds)^2))/RangeOfRatings

########Question d ##########

#We make sure to treat the genre of the song and the year as factors/categorical variables

train_merge <- merge(train, Songs, by = 'songID')
train_merge$year <- as.factor(train_merge$year)
train_merge$genre <- as.factor(train_merge$genre)

test_merge <- merge(test, Songs, by = 'songID')
test_merge$year <- as.factor(test_merge$year)
test_merge$genre <- as.factor(test_merge$genre)

valB_merge <- merge(valB, Songs, by = 'songID')
valB_merge$year <- as.factor(valB_merge$year)
valB_merge$genre <- as.factor(valB_merge$genre)

#we first train a random forest model
rf.mod <- ranger(rating ~ year + genre, 
                 data = train_merge, 
                 num.trees = 500,
                 verbose = FALSE)

#we evaluate the performance of the random forest model
pred.rf <- predict(rf.mod, data = test_merge)
pred.rf <- pred.rf$predictions
MAE_rf = mean(abs(pred.rf - test_merge$rating))/RangeOfRatings
RMSE_rf = sqrt(mean((pred.rf - test_merge$rating)^2))/RangeOfRatings
OSR2_rf = OSR2(pred.rf, train_merge$rating, test_merge$rating)
MAE_rf 
RMSE_rf
OSR2_rf

#we train a cart model (we cross validate to choose cp)

CART.mod <- train(rating ~ year + genre,
                  data = train_merge,
                  method = "rpart",
                  tuneGrid = data.frame(cp=seq(0, 0.05, 0.001)),
                  trControl = trainControl(method="cv", number = 5), 
                  metric = "RMSE"
)
# look at the cross validation results, stored as a data-frame
CART.mod$results 
CART.mod
ggplot(CART.mod$results, aes(x=cp, y= RMSE)) + geom_point(size=3) +
  xlab("Complexity Parameter (cp)") + geom_line()

#we evaluate the performance of cart model
valB_merge_mm = as.data.frame(model.matrix(rating ~ . + 0, data=valB_merge))
test_merge_mm = as.data.frame(model.matrix(rating ~ . + 0, data=test_merge))

preds.CART <- predict(CART.mod$finalModel, newdata = test_merge_mm)

MAE_CART = mean(abs(preds.CART - test_merge$rating))/RangeOfRatings
RMSE_CART = sqrt(mean((preds.CART - test_merge$rating)^2))/RangeOfRatings
OSR2_CART = OSR2(preds.CART, train_merge$rating, test_merge$rating)
MAE_CART
RMSE_CART
OSR2_CART

# Blending

val.preds.cf <- impute(mod_2, valB$userID, valB$songID)
val.preds.lm <- alpha[valB$userID] + beta[valB$songID]
val.preds.rf <- predict(rf.mod, data = valB_merge)$predictions
val.preds.CART <- predict(CART.mod$finalModel, newdata = valB_merge_mm)

# Build validation set data frame
val.final_df = data.frame(rating = valB$rating, cf_preds = val.preds.cf, 
                          lm_preds = val.preds.lm, rf_preds = val.preds.rf, CART_preds = val.preds.CART)

# Train blended model
blended.mod = lm(rating ~ . -1, data = val.final_df)
summary(blended.mod)

# Get predictions on test set
test.preds.cf <- impute(mod_2, test$userID, test$songID)
test.preds.lm <- alpha[test$userID] + beta[test$songID]
test.preds.rf <- pred.rf
test.preds.CART <- preds.CART

test.final_df = data.frame(rating = test$rating, cf_preds = test.preds.cf, 
                           lm_preds = test.preds.lm, rf_preds = test.preds.rf, CART_preds = test.preds.CART)

test.preds.blended <- predict(blended.mod, newdata = test.final_df)

MAE_blended = mean(abs(test.preds.blended - test$rating))/RangeOfRatings
RMSE_blended = sqrt(mean((test.preds.blended - test$rating)^2))/RangeOfRatings
OSR2_blended = OSR2(test.preds.blended, train$rating, test$rating)
MAE_blended
RMSE_blended
OSR2_blended

