
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

#########################
## 제6장 성능평가 방법 ##
#########################

##############
## 6.3 사례 ##
##############

## 홀드아웃법

library(mlbench)
data(PimaIndiansDiabetes)
str(PimaIndiansDiabetes)

nrow(PimaIndiansDiabetes)
size <- round(nrow(PimaIndiansDiabetes)*c(0.5, 0.25, 0.25))
size

set.seed(123)
index <- order(runif(nrow(PimaIndiansDiabetes)))
index

pima.train <- PimaIndiansDiabetes[index[1:384],]
pima.validation <- PimaIndiansDiabetes[index[385:576],]
pima.test <- PimaIndiansDiabetes[index[577:768],]
nrow(pima.train)
nrow(pima.validation)
nrow(pima.test)

library(rsample)
set.seed(123)
split <- initial_validation_split(data=PimaIndiansDiabetes, 
                                  prop=c(0.5, 0.25), strata=diabetes)

pima.train <- training(split)
pima.validation <- validation(split)
pima.test <- testing(split)

nrow(pima.train)
nrow(pima.validation)
nrow(pima.test)
prop.table(table(pima.train$diabetes))
prop.table(table(pima.validation$diabetes))
prop.table(table(pima.test$diabetes))
prop.table(table(PimaIndiansDiabetes$diabetes))

library(class)
k <- 20
result <- numeric(k)
for (j in 1:k) {
  knn.fit <- knn(train=pima.train[c(1:8)], test=pima.validation[c(1:8)], 
                 cl=pima.train$diabetes, k=j)
  result[j] <- mean(pima.validation$diabetes==knn.fit)
}

knn.result <- data.frame(k=1:k, accuracy=result)
knn.result
summary(knn.result$accuracy)
which.max(knn.result$accuracy)
knn.result$accuracy[which.max(knn.result$accuracy)]

knn.fit <- knn(train=pima.train[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.train$diabetes, k=which.max(knn.result$accuracy))
mean(pima.test$diabetes==knn.fit)

## 부트스트래핑

library(mlbench)
library(caret)
data(PimaIndiansDiabetes)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes$diabetes, p=0.7, list=FALSE)
pima.trainvalid <- PimaIndiansDiabetes[index,]
pima.test <- PimaIndiansDiabetes[-index,]
nrow(pima.trainvalid)
nrow(pima.test)
nrow(PimaIndiansDiabetes)

library(rsample)
set.seed(123)
bs <- bootstraps(data=pima.trainvalid, times=10, strata=diabetes)
class(bs)
bs

bs$splits

library(tibble)
pima.train1 <- as_tibble(bs$splits[[1]])
pima.train1

index.validation1 <- complement(bs$splits[[1]])
pima.validation1 <- as_tibble(pima.trainvalid[index.validation1,])
pima.validation1

lapply(bs$splits, function(x) as_tibble(x))
lapply(bs$splits, function(x) {
  index.validation <- complement(x)
  as_tibble(pima.trainvalid[index.validation,])
})

sapply(bs$splits, function(x) {
  dat <- as_tibble(x)$diabetes
  mean(dat=="pos")
})
sapply(bs$splits, function(x) {
  index.validation <- complement(x)
  dat <- as_tibble(pima.trainvalid[index.validation,])$diabetes
  mean(dat=="pos")
})
prop.table(table(PimaIndiansDiabetes$diabetes))["pos"]

library(class)
k <- 20
run <- 10
trial.sum <- numeric(k)
trial.n <- numeric(k)
for (i in 1:run) {
  pima.train <- as_tibble(bs$splits[[i]])
  index.validation <- complement(bs$splits[[i]])
  pima.validation <- as_tibble(pima.trainvalid[index.validation,])
  validation.size <- nrow(pima.validation)
  for (j in 1:k) {
    knn.fit <- knn(train=pima.train[c(1:8)], test=pima.validation[c(1:8)], 
                   cl=pima.train$diabetes, k=j)
    trial.sum[j] <- trial.sum[j] + sum(pima.validation$diabetes==knn.fit)
    trial.n[j] <- trial.n[j] + validation.size
  }
}
knn.result <- data.frame(k=1:k, accuracy=trial.sum/trial.n)
knn.result
summary(knn.result$accuracy)
which.max(knn.result$accuracy)

knn.fit <- knn(train=pima.trainvalid[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.trainvalid$diabetes, k=which.max(knn.result$accuracy))
mean(pima.test$diabetes==knn.fit)

library(caret)
set.seed(123)
caret.boot <- train(diabetes ~ ., data=pima.trainvalid, 
                    metric="Accuracy",
                    method="knn",
                    trControl=trainControl(method="boot", number=10),
                    tuneGrid=expand.grid(k=seq(from=1, to=20, by=1)))
class(caret.boot)
caret.boot

library(dplyr)
caret.boot$resample  |> 
  arrange(Resample)

caret.boot$resample |> 
  summarise(AvgAccuracy=mean(Accuracy))
caret.boot$bestTune

caret.boot$results |> 
  slice_max(order_by=Accuracy)

getTrainPerf(caret.boot)

knn.fit <- knn(train=pima.trainvalid[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.trainvalid$diabetes, k=caret.boot$bestTune)
mean(pima.test$diabetes==knn.fit)

## 교차검증

# k-폴드 교차검증 (k-fold CV)

library(mlbench)
library(caret)
data(PimaIndiansDiabetes)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes$diabetes, p=0.7, list=FALSE)
pima.trainvalid <- PimaIndiansDiabetes[index,]
pima.test <- PimaIndiansDiabetes[-index,]
nrow(pima.trainvalid)
nrow(pima.test)
nrow(PimaIndiansDiabetes)

library(rsample)
set.seed(123)
folds <- vfold_cv(data=pima.trainvalid, v=10, strata=diabetes)
class(folds)
folds

folds$splits

library(tibble)
train.kf <- lapply(folds$splits, function(x) as_tibble(x))
validation.kf <- lapply(folds$splits, function(x) {
  index.valid <- complement(x)
  as_tibble(pima.trainvalid[index.valid,])
})
sapply(train.kf, nrow)
sapply(validation.kf, nrow)

library(caret)
set.seed(456)
folds <- createFolds(y=pima.trainvalid$diabetes, k=10)
class(folds)
str(folds)

train.kf <- lapply(folds, function(x) as_tibble(pima.trainvalid[-x,]))
validation.kf <- lapply(folds, function(x) as_tibble(pima.trainvalid[x,]))
sapply(train.kf, nrow)
sapply(validation.kf, nrow)

library(class)
k <- 20
run <- 10
trial.sum <- numeric(k)
trial.n <- numeric(k)
for (i in 1:run) {
  pima.train <- train.kf[[i]]
  pima.validation <- validation.kf[[i]]
  validation.size <- nrow(pima.validation)
  for (j in 1:k) {
    knn.fit <- knn(train=pima.train[c(1:8)], test=pima.validation[c(1:8)], 
                   cl=pima.train$diabetes, k=j)
    trial.sum[j] <- trial.sum[j] + sum(pima.validation$diabetes==knn.fit)
    trial.n[j] <- trial.n[j] + validation.size
  }
}
knn.result <- data.frame(k=1:k, accuracy=trial.sum/trial.n)
knn.result
summary(knn.result$accuracy)
which.max(knn.result$accuracy)

knn.fit <- knn(train=pima.trainvalid[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.trainvalid$diabetes, k=which.max(knn.result$accuracy))
mean(pima.test$diabetes==knn.fit)

library(caret)
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.trainvalid, 
                  metric="Accuracy",
                  method="knn",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=expand.grid(k=seq(from=1, to=20, by=1)))
class(caret.cv)
caret.cv

library(dplyr)
caret.cv$resample |> 
  arrange(Resample)

caret.cv$resample |> 
  summarise(AvgAccuracy=mean(Accuracy))
caret.cv$bestTune

caret.cv$results |> 
  slice_max(order_by=Accuracy)

knn.fit <- knn(train=pima.trainvalid[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.trainvalid$diabetes, k=caret.cv$bestTune)
mean(pima.test$diabetes==knn.fit)

# LOOCV (Leave-One-Out CV)

library(mlbench)
library(caret)
data(PimaIndiansDiabetes)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes$diabetes, p=0.7, list=FALSE)
pima.trainvalid <- PimaIndiansDiabetes[index,]
pima.test <- PimaIndiansDiabetes[-index,]
nrow(pima.trainvalid)
nrow(pima.test)
nrow(PimaIndiansDiabetes)

library(caret)
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.trainvalid, 
                  metric="Accuracy",
                  method="knn",
                  trControl=trainControl(method="LOOCV"),
                  tuneGrid=expand.grid(k=seq(from=1, to=20, by=1)))
caret.cv

library(dplyr)
caret.cv$results |> 
  slice_max(order_by=Accuracy)
caret.cv$bestTune

library(class)
knn.fit <- knn(train=pima.trainvalid[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.trainvalid$diabetes, k=caret.cv$bestTune)
mean(pima.test$diabetes==knn.fit)

# LGOCV (Leave-Group-Out CV)

library(mlbench)
library(caret)
data(PimaIndiansDiabetes)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes$diabetes, p=0.7, list=FALSE)
pima.trainvalid <- PimaIndiansDiabetes[index,]
pima.test <- PimaIndiansDiabetes[-index,]
nrow(pima.trainvalid)
nrow(pima.test)
nrow(PimaIndiansDiabetes)

library(caret)
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.trainvalid, 
                  metric="Accuracy",
                  method="knn",
                  trControl=trainControl(method="LGOCV", p=0.9, number=10),
                  tuneGrid=expand.grid(k=seq(from=1, to=20, by=1)))
caret.cv

library(dplyr)
caret.cv$resample |> 
  arrange(Resample)

caret.cv$resample |> 
  summarise(AvgAccuracy=mean(Accuracy))
caret.cv$bestTune

caret.cv$results |> 
  slice_max(order_by=Accuracy)

knn.fit <- knn(train=pima.trainvalid[c(1:8)], test=pima.test[c(1:8)], 
               cl=pima.trainvalid$diabetes, k=caret.cv$bestTune)
mean(pima.test$diabetes==knn.fit)
