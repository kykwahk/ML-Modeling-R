
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

########################
## 제4장 데이터전처리 ##
########################

##############
## 4.1 개요 ##
##############

library(modeldata)
str(ames)

library(rsample)
set.seed(123)
split <- initial_split(data=ames, prop=0.7)
ames.train <- training(split)
ames.test <- testing(split)
dim(ames.train)
dim(ames.test)

##################
## 4.2 변수제거 ##
##################

library(tidyverse)
library(caret)
nearZeroVar(ames.train, saveMetrics=TRUE)
nearZeroVar(ames.train, saveMetrics=TRUE) |> 
  rownames_to_column(var="predictor") |> 
  filter(zeroVar | nzv)

# [그림 4-1]
windows(width=10, height=10)
library(corrplot)
corr.mat <- cor(select(ames.train, where(is.numeric) & -Sale_Price))
corrplot(corr.mat, method="color", type="upper", order="hclust")

corr.df <- as.data.frame(as.table(corr.mat), stringsAsFactors=FALSE)
colnames(corr.df) <- c("var1", "var2", "corr")
head(corr.df)

corr.flat <- corr.df |> 
  filter(var1 != var2) |> 
  mutate(pair=paste(pmin(var1, var2), pmax(var1, var2), sep="-")) |> 
  distinct(pair, .keep_all=TRUE) |> 
  select(-pair)
head(corr.flat)

slice_max(corr.flat, order_by=abs(corr), n=10)

arrange(corr.flat, desc(abs(corr))) |> 
  filter(abs(corr) > 0.75)

library(caret)
findCorrelation(corr.mat, cutoff=0.75, names=TRUE)

library(recipes)
recipe <- recipe(Sale_Price ~ ., data=ames.train)
class(recipe)
recipe

summary(recipe)
print(summary(recipe), n=Inf)

?recipes::selections
?tidyselect
?dplyr::select

recipe.step <- recipe |> 
  step_zv(all_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_corr(all_numeric_predictors(), threshold=0.75)
recipe.step
class(recipe.step)

tidy(recipe.step)

tidy(recipe.step, number=3)

recipe.step2 <- recipe |> 
  step_zv(all_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_corr(all_numeric_predictors(), threshold=0.75, id="my.corr")
tidy(recipe.step2)
tidy(recipe.step2, number=3)
tidy(recipe.step2, id="my.corr")

prepare <- prep(recipe.step, training=ames.train)
prepare
class(prepare)
summary(prepare)
print(summary(prepare), n=Inf)

tidy(prepare)

tidy(prepare, number=2)
tidy(prepare, number=3)

bake.train <- bake(prepare, new_data=ames.train)
bake.train
class(bake.train)

bake.test <- bake(prepare, new_data=ames.test)
bake.test
class(bake.test)

bake.train <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_zv(all_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_corr(all_numeric_predictors(), threshold=0.75) |> 
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_zv(all_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_corr(all_numeric_predictors(), threshold=0.75) |> 
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

nearZeroVar(bake.train, saveMetrics=FALSE)
nearZeroVar(bake.test, saveMetrics=FALSE)

corr.mat.train <- cor(select(bake.train, where(is.numeric) & -Sale_Price))
sum(abs(corr.mat.train[upper.tri(corr.mat.train)]) > 0.75)
corr.mat.test <- cor(select(bake.test, where(is.numeric) & -Sale_Price))
sum(abs(corr.mat.test[upper.tri(corr.mat.test)]) > 0.75)

findCorrelation(corr.mat.train, cutoff=0.75)
findCorrelation(corr.mat.test, cutoff=0.75)

##################
## 4.3 결측처리 ##
##################

## 결측식별

library(AmesHousing)
dim(ames_raw)
names(ames_raw)

sum(is.na(ames_raw))

sum(complete.cases(ames_raw))

library(naniar)
miss_var_summary(ames_raw)

prop_complete_case(ames_raw)

# [그림 4-3]
library(VIM)
windows(width=12.0, height=11.0)
aggr(ames_raw, sortVar=TRUE, prop=FALSE, numbers=TRUE, combined=TRUE, 
     cex.axis=0.7, bars=FALSE)

# [그림 4-4]
library(visdat)
windows(width=12.0, height=8.0)
vis_miss(ames_raw, cluster=TRUE)

## 결측대체: 정보성 결측

library(rsample)
set.seed(123)
split <- initial_split(data=ames_raw, prop=0.7)
ames_raw.train  <- training(split)
ames_raw.test <- testing(split)
dim(ames_raw.train)
dim(ames_raw.test)

filter(ames_raw.train, `Garage Cars`==0 & `Garage Area`==0) |> 
  select(starts_with("Garage"))

filter(ames_raw.train, `Garage Cars`==0 & `Garage Area`==0) |> 
  mutate(across(starts_with("Garage") & where(~(is.character(.x) | is.factor(.x))), ~"None")) |> 
  mutate(across(starts_with("Garage") & where(is.numeric), ~0)) |> 
  select(starts_with("Garage"))

recipe.step <- recipe(SalePrice ~ ., data=ames_raw.train, strings_as_factors=FALSE) |> 
  step_mutate(across(starts_with("Garage") & where(~(is.character(.x) | is.factor(.x))),
                     ~ifelse(is.na(.x) & `Garage Cars`==0 & `Garage Area`==0, 
                             "None", as.character(.x)))) |> 
  step_mutate(across(starts_with("Garage") & where(is.numeric),
                     ~ifelse(is.na(.x) & `Garage Cars`==0 & `Garage Area`==0, 
                             0, as.numeric(.x))))
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames_raw.train) |> 
  bake(new_data=ames_raw.train)
filter(bake.train, `Garage Cars`==0 & `Garage Area`==0) |> 
  select(starts_with("Garage"))

bake.test <- recipe.step |>  
  prep(training=ames_raw.train) |> 
  bake(new_data=ames_raw.test)
filter(bake.test, `Garage Cars`==0 & `Garage Area`==0) |> 
  select(starts_with("Garage"))

## 결측대체: 무작위 결측 - 평균/중위수/최빈값 대체

recipe.step <- recipe(SalePrice ~ ., data=ames_raw.train, strings_as_factors=FALSE) |> 
  update_role(PID, new_role="ID") |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors())
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames_raw.train) |> 
  bake(new_data=ames_raw.train)
bake.test <- recipe.step |>  
  prep(training=ames_raw.train) |> 
  bake(new_data=ames_raw.test)

anyNA(bake.train)
anyNA(bake.test)

head(ames_raw.train$`Lot Frontage`)
median(ames_raw.train$`Lot Frontage`, na.rm=TRUE)
head(bake.train$`Lot Frontage`)

head(ames_raw.test$`Lot Frontage`)
head(bake.test$`Lot Frontage`)

head(ames_raw.train$Fence)
table(ames_raw.train$Fence)
head(bake.train$Fence)

head(ames_raw.test$Fence)
head(bake.test$Fence)

## 결측대체: k-NN 대체

recipe.step <- recipe(SalePrice ~ ., data=ames_raw.train, strings_as_factors=FALSE) |> 
  update_role(PID, new_role="ID") |>
  step_impute_knn(all_predictors(), neighbors=7) 
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames_raw.train) |> 
  bake(new_data=ames_raw.train)
bake.test <- recipe.step |>  
  prep(training=ames_raw.train) |> 
  bake(new_data=ames_raw.test)

anyNA(bake.train)
anyNA(bake.test)

head(ames_raw.train$`Lot Frontage`)
head(bake.train$`Lot Frontage`)
head(ames_raw.test$`Lot Frontage`)
head(bake.test$`Lot Frontage`)

head(ames_raw.train$Fence)
head(bake.train$Fence)
head(ames_raw.test$Fence)
head(bake.test$Fence)

###############################
## 4.4 변수변환: 연속형 변수 ##
###############################

library(modeldata)
str(ames)
library(rsample)
set.seed(123)
split <- initial_split(data=ames, prop=0.7)
ames.train <- training(split)
ames.test <- testing(split)

## 정규성변환

library(e1071)
ames.train |> 
  summarise(across(where(is.numeric), ~skewness(.x, type=1))) |> 
  pivot_longer(everything(), names_to="variable", values_to="skewness") |> 
  filter(skewness > 2 | skewness < -2)

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_YeoJohnson(all_numeric(), -all_outcomes())
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

tidy(prep(recipe.step, training=ames.train), number=1)

head(ames.train$Gr_Liv_Area)
head(bake.train$Gr_Liv_Area)
head(ames.test$Gr_Liv_Area)
head(bake.test$Gr_Liv_Area)

# [그림 4-5]
windows(width=8.0, height=4.5)
trans.no <- tibble(var="Gr_Liv_Area", val=ames.train$Gr_Liv_Area, trans="no")
trans.yes <- tibble(var="Gr_Liv_Area", val=bake.train$Gr_Liv_Area, trans="yes")
transform <- bind_rows(trans.no, trans.yes) |> 
  mutate(trans=factor(trans, levels=c("no", "yes"), 
                      labels=c("No Normality Transformation", "Normality Transformation")))
transform
ggplot(data=transform, aes(x=val)) +
  geom_histogram(color="dimgray", fill="mistyrose") +
  facet_wrap(vars(trans), scales="free_x") +
  labs(x=transform$var, y="",
       title="Distribution Transformation") +
  theme_minimal() +
  theme(strip.text=element_text(face="bold", size=10, color="dimgray"),
        plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank()) 

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_log(all_outcomes())
recipe.step
bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

head(ames.train$Sale_Price)
head(bake.train$Sale_Price)

y <- log(10)
y
exp(y)

library(forecast)
y <- BoxCox(10, lambda=0.5)
y
InvBoxCox(y, lambda=0.5)

library(VGAM)
y <- yeo.johnson(10, lambda=0.5)
y
yeo.johnson(y, lambda=0.5, inverse=TRUE)

## 피처 스케일링

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_center(all_numeric(), -all_outcomes()) |> 
  step_scale(all_numeric(), -all_outcomes())
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

var <- c("Total_Bsmt_SF", "Garage_Area", "Gr_Liv_Area")
apply(ames.train[var], 2, mean, na.rm=TRUE); apply(ames.train[var], 2, sd, na.rm=TRUE)
apply(bake.train[var], 2, mean, na.rm=TRUE); apply(bake.train[var], 2, sd, na.rm=TRUE)

apply(ames.test[var], 2, mean, na.rm=TRUE); apply(ames.test[var], 2, sd, na.rm=TRUE)
apply(bake.test[var], 2, mean, na.rm=TRUE); apply(bake.test[var], 2, sd, na.rm=TRUE)

# [그림 4-6]
windows(width=8.0, height=4.5)
trans.no <- ames.train |> 
  select(c("Total_Bsmt_SF", "Garage_Area", "Gr_Liv_Area")) |> 
  pivot_longer(cols=everything(), names_to="var", values_to="val") |> 
  mutate(trans="no")
trans.yes <- bake.train |> 
  select(c("Total_Bsmt_SF", "Garage_Area", "Gr_Liv_Area")) |> 
  pivot_longer(cols=everything(), names_to="var", values_to="val") |> 
  mutate(trans="yes")
transform <- bind_rows(trans.no, trans.yes) |> 
  mutate(trans=factor(trans, levels=c("no", "yes"), 
                      labels=c("No Z Score Standardization", "Z Score Standardization")))
transform
ggplot(data=transform, aes(x=var, y=val)) +
  geom_jitter(pch=21, color="dimgray", fill="mistyrose") +
  geom_boxplot(color="black", fill="salmon", alpha=0.5, 
               outlier.shape=21, outlier.color="dimgray", outlier.fill="mistyrose") +
  facet_wrap(vars(trans), scales="free_x") +
  labs(x="Predictor", y="Value",
       title="Feature Scaling") +
  coord_flip() +
  theme_minimal() +
  theme(strip.text=element_text(face="bold", size=10, color="dimgray"),
        plot.title=element_text(face="bold"),
        axis.title=element_text(face="bold"),
        axis.text=element_text(size=8.5, face="bold"),
        axis.line=element_line(color="gray"),
        panel.grid.minor=element_blank()) 

###############################
## 4.5 변수변환: 범주형 변수 ##
###############################

## 럼핑(lumping)

count(ames.train, Neighborhood) |> 
  mutate(ratio=n/sum(n)) |> 
  arrange(n)

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_other(Neighborhood, threshold=0.01, other="Other")
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

count(bake.train, Neighborhood) |> 
  arrange(n) 
count(bake.test, Neighborhood) |> 
  arrange(n) 

tidy(prep(recipe.step, training=ames.train), number=1)

## 원-핫인코딩과 더미인코딩

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |>
  step_dummy(all_nominal(), one_hot=TRUE)
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

tail(ames.train$Roof_Style)
count(ames.train, Roof_Style)

tail(select(bake.train, starts_with("Roof_Style")))
select(bake.train, starts_with("Roof_Style")) |> 
  summarise(across(everything(), sum)) |> 
  pivot_longer(everything(), names_to="variable", values_to="n")

tidy(prep(recipe.step, training=ames.train), number=1)

## 레이블 인코딩

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |>
  step_integer(Bldg_Type)
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)

head(ames.train$Bldg_Type)
count(ames.train, Bldg_Type)
head(bake.train$Bldg_Type)
count(bake.train, Bldg_Type)

tidy(prep(recipe.step, training=ames.train), number=1)$value

##################
## 4.6 차원축소 ##
##################

recipe.step <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_center(all_numeric(), -all_outcomes()) |> 
  step_scale(all_numeric(), -all_outcomes()) |> 
  step_pca(all_numeric(), -all_outcomes(), threshold=0.75)
recipe.step

bake.train <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe.step |>  
  prep(training=ames.train) |> 
  bake(new_data=ames.test)
names(bake.train)
names(bake.test)

bake.train <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_center(all_numeric(), -all_outcomes()) |> 
  step_scale(all_numeric(), -all_outcomes()) |> 
  step_pca(all_numeric(), -all_outcomes(), num_comp=5) |> 
  prep(training=ames.train) |> 
  bake(new_data=ames.train)
bake.test <- recipe(Sale_Price ~ ., data=ames.train) |> 
  step_center(all_numeric(), -all_outcomes()) |> 
  step_scale(all_numeric(), -all_outcomes()) |> 
  step_pca(all_numeric(), -all_outcomes(), num_comp=5) |> 
  prep(training=ames.train) |> 
  bake(new_data=ames.test)
names(bake.train)
names(bake.test)

##################
## 4.7 변수생성 ##
##################

library(nycflights13) 
str(flights)
names(flights)

library(rsample)
set.seed(123)
split <- initial_split(data=flights, prop=0.7)
flights.train <- training(split)
flights.test <- testing(split)
dim(flights.train)
dim(flights.test)

recipe <- recipe(arr_delay ~ ., data=flights.train)
recipe

library(timeDate)
recipe.step <- recipe |> 
  update_role(flight, tailnum, new_role="ID") |> 
  step_date(time_hour, features=c("dow", "month")) |>               
  step_holiday(time_hour, 
               holidays=listHolidays("US"), 
               keep_original_cols=FALSE)
recipe.step

bake.train <- recipe.step |>  
  prep(training=flights.train) |> 
  bake(new_data=flights.train)
bake.test <- recipe.step |>  
  prep(training=flights.train) |> 
  bake(new_data=flights.test)
str(bake.train)

select(bake.train, starts_with("time_hour"))
colnames(select(bake.train, starts_with("time_hour")))

tidy(prep(recipe.step, training=flights.train), number=1)
tidy(prep(recipe.step, training=flights.train), number=2)

##############
## 4.8 사례 ##
##############

library(modeldata)
str(credit_data)

library(rsample)
set.seed(123)
split <- initial_split(data=credit_data, prop=0.7)
credit_data.train  <- training(split)
credit_data.test   <- testing(split)
dim(credit_data.train)
dim(credit_data.test)

library(skimr)
skim(credit_data.train)

library(caret)
nearZeroVar(credit_data.train, saveMetrics=TRUE)

library(recipes)
recipe.step <- recipe(Status ~ ., data=credit_data.train) |> 
  step_zv(all_predictors()) |> 
  step_nzv(all_predictors()) |> 
  step_impute_knn(all_predictors(), neighbors=5) |>
  step_center(all_numeric(), -all_outcomes()) |> 
  step_scale(all_numeric(), -all_outcomes()) |> 
  step_pca(all_numeric(), -all_outcomes(), threshold=0.75) |> 
  step_dummy(all_nominal(), -all_outcomes(), one_hot=TRUE)
recipe.step

prepare <- prep(recipe.step, training=credit_data.train)
prepare

tidy(prepare)

bake.train <- bake(prepare, new_data=credit_data.train)
bake.test <- bake(prepare, new_data=credit_data.test)
bake.train
bake.test
