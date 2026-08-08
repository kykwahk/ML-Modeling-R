
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

####################
## 제3장 모델개발 ##
####################

#######################
## 3.2 데이터셋 분할 ##
#######################

## 단순무작위표본추출

library(MASS)
str(Boston)

set.seed(123)
index <- sample(x=nrow(Boston), size=round(0.7*nrow(Boston)))
Boston.train1 <- Boston[index,]
Boston.test1 <- Boston[-index,]
dim(Boston.train1)
dim(Boston.test1)

library(rsample)
set.seed(456)
split <- initial_split(data=Boston, prop=0.7)
Boston.train2 <- training(split)
Boston.test2 <- testing(split)
dim(Boston.train2)
dim(Boston.test2)

# [그림 3-2]
library(tidyverse)
house.train1 <- tibble(price=Boston.train1$medv, dataset="train")
house.test1 <- tibble(price=Boston.test1$medv, dataset="test")
house1 <- bind_rows(house.train1, house.test1) %>% 
  mutate(dataset=factor(dataset, levels=c("train", "test")))
house1
p1 <- ggplot(data=house1, aes(x=price, color=dataset, lty=dataset)) +
  geom_density(lwd=1.1) +
  labs(title="base", x="Price", y="Density") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank(),
        legend.position="none") 
p1
house.train2 <- tibble(price=Boston.train2$medv, dataset="train")
house.test2 <- tibble(price=Boston.test2$medv, dataset="test")
house2 <- bind_rows(house.train2, house.test2) %>% 
  mutate(dataset=factor(dataset, levels=c("train", "test")))
house2
p2 <- ggplot(data=house2, aes(x=price, color=dataset, lty=dataset)) +
  geom_density(lwd=1.1) +
  labs(title="rsample", x="Price", y="Density") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank(),
        legend.position="none") 
p2
windows(width=7.0, height=5.5)
library(patchwork)
p1 + p2 + 
  plot_layout(guide="collect") & theme(legend.position="bottom") & 
  labs(color="Dataset", lty="Dataset")

## 층화무작위표본추출

library(modeldata)
data(credit_data)
str(credit_data)
prop.table(table(credit_data$Status))

library(caret)
set.seed(123)
index <- createDataPartition(y=credit_data$Status, p=0.7, list=FALSE)
credit.train1 <- credit_data[index,]
credit.test1 <- credit_data[-index,]
dim(credit.train1)
dim(credit.test1)
table(credit.train1$Status)
table(credit.test1$Status)
prop.table(table(credit.train1$Status))
prop.table(table(credit.test1$Status))

library(rsample)
set.seed(456)
split <- initial_split(data=credit_data, prop=0.7, strata="Status")
credit.train2 <- training(split)
credit.test2 <- testing(split)
dim(credit.train2)
dim(credit.test2)
table(credit.train2$Status)
table(credit.test2$Status)
prop.table(table(credit.train2$Status))
prop.table(table(credit.test2$Status))

# [그림 3-3]
creditc.train1 <- tibble(status=credit.train1$Status, dataset="train")
creditc.test1 <- tibble(status=credit.test1$Status, dataset="test")
credit1 <- bind_rows(creditc.train1, creditc.test1) %>% 
  mutate(dataset=factor(dataset, levels=c("train", "test")))
credit1
p1 <- ggplot(data=credit1, aes(x=dataset, fill=status)) +
  geom_bar(position="fill", width=0.7) +
  scale_fill_manual(values=c("salmon", "cornflowerblue")) +
  labs(title="caret", x="Dataset", y="Proportion") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank(),
        legend.position="none") 
p1
creditc.train2 <- tibble(status=credit.train2$Status, dataset="train")
creditc.test2 <- tibble(status=credit.test2$Status, dataset="test")
credit2 <- bind_rows(creditc.train2, creditc.test2) %>% 
  mutate(dataset=factor(dataset, levels=c("train", "test")))
credit2
p2 <- ggplot(data=credit2, aes(x=dataset, fill=status)) +
  geom_bar(position="fill", width=0.7) +
  scale_fill_manual(values=c("salmon", "cornflowerblue")) +
  labs(title="rsample", x="Dataset", y="Proportion") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank(),
        legend.position="none") 
p2
windows(width=7.0, height=5.5)
library(patchwork)
p1 + p2 + 
  plot_layout(guide="collect") & theme(legend.position="bottom") & labs(fill="Status")

## 클래스불균형

library(mlbench)
data(PimaIndiansDiabetes)
str(PimaIndiansDiabetes)
table(PimaIndiansDiabetes$diabetes)
prop.table(table(PimaIndiansDiabetes$diabetes))

library(dplyr)
library(smotefamily) 
set.seed(123)
PimaIndiansDiabetes.new <- SMOTE(X=select(PimaIndiansDiabetes, -diabetes), 
                                 target=PimaIndiansDiabetes$diabetes, K=3, dup_size=1)
str(PimaIndiansDiabetes.new)
PimaIndiansDiabetes.new <- PimaIndiansDiabetes.new$data |> 
  rename(diabetes=class)
table(PimaIndiansDiabetes.new$diabetes)
prop.table(table(PimaIndiansDiabetes.new$diabetes))

# [그림 3-5]
p1 <- ggplot(data=PimaIndiansDiabetes, aes(x=mass, y=insulin, fill=diabetes)) +
  geom_point(pch=21, color="black", alpha=0.5) +
  labs(x="BMI", y="Insulin",
       title="Original Data", fill="diabetes") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank(),
        legend.position="none") 
p2 <- ggplot(data=PimaIndiansDiabetes.new, aes(x=mass, y=insulin, fill=diabetes)) +
  geom_point(pch=21, color="black", alpha=0.5) +
  labs(x="BMI", y="Insulin",
       title="SMOTE Data", fill="diabetes") +
  theme_minimal() +
  theme(plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank(),
        legend.position="none") 
windows(width=7.0, height=4.5)
library(patchwork)
p1 + p2 + 
  plot_layout(guide="collect") & theme(legend.position="bottom")

############################
## 3.6 하이퍼파라미터튜닝 ##
############################

str(iris)
levels(iris$Species)

library(caret)
library(class)
k <- 20
run <- 100
trial.sum <- numeric(k)
trial.n <- numeric(k)
for (i in 1:run) {
  index <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
  iris.train <- iris[index,]
  iris.test <- iris[-index,]
  test.size <- nrow(iris.test)
  for (j in 1:k) {
    knn.fit <- knn(train=iris.train[c(1:4)], test=iris.test[c(1:4)], 
                   cl=iris.train$Species, k=j)
    trial.sum[j] <- trial.sum[j] + sum(iris.test$Species==knn.fit)
    trial.n[j] <- trial.n[j] + test.size
    }
  }

knn.cv <- data.frame(k=1:k, accuracy=trial.sum/trial.n)
knn.cv
summary(knn.cv$accuracy)
which.max(knn.cv$accuracy)

# [그림 3-12]
windows(width=7.0, height=4.0)
library(ggplot2)
ggplot(knn.cv, aes(x=k, y=accuracy)) +
  geom_line(color="dimgray", lwd=1) +
  geom_point(shape=21, color="black", fill="red", size=2.5, stroke=1) +
  labs(x="k", y="Accuracy") +
  scale_x_continuous(breaks=seq(from=0, to=20, by=5)) +
  theme_bw()

set.seed(123)
index <- createDataPartition(y=iris$Species, p=0.7, list=FALSE)
iris.train <- iris[index,]
iris.test <- iris[-index,]
dim(iris.train)
dim(iris.test)

caret.cv <- train(Species ~ ., data=iris.train, method="knn",
                  trControl=trainControl(method="cv", number=10),
                  tuneGrid=expand.grid(k=seq(from=1, to=20, by=1)))
caret.cv

# [그림 3-13]
windows(width=7.0, height=4.0)
ggplot(caret.cv) +
  geom_line(color="dimgray", lwd=1) +
  geom_point(shape=21, color="black", fill="blue", size=2.5, stroke=1) +
  labs(x="k", y="Accuracy") +
  scale_x_continuous(breaks=seq(from=0, to=50, by=5)) +
  theme_bw()
