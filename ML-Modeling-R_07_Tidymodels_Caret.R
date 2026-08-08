
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

############################
## 제7장 통합 모델링 도구 ##
############################

####################
## 7.1 Tidymodels ##
####################

library(tidymodels)
tidymodels_prefer()

## 모델구축

decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("classification")

get_from_env("decision_tree")

decision_tree(tree_depth=20, min_n=3, cost_complexity=0.05) |> 
  set_engine("rpart") |> 
  set_mode("classification") |> 
  translate()

args("decision_tree")
get_from_env("decision_tree_args")

show_model_info("decision_tree")

library(MASS)
str(Boston)

set.seed(123)
split <- initial_split(data=Boston, prop=0.7)
class(split)
Boston.train <- training(split)
Boston.test <- testing(split)
dim(Boston.train)
dim(Boston.test)

tidy.mod <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")
class(tidy.mod)
tidy.mod

tidy.fit <- fit(tidy.mod, medv ~ crim + rm, data=Boston.train)
class(tidy.fit)
tidy.fit

class(tidy.fit$fit)
tidy.fit$fit

tidy.fit |> 
  extract_fit_engine()

# [그림 7-2]
library(rattle)
windows(width=7.0, height=5.5)
tidy.fit |> 
  extract_fit_engine() |> 
  fancyRpartPlot(sub=NULL)

tidy.pred <- predict(tidy.fit, new_data=Boston.test)
tidy.pred

tidy.res <- bind_cols(tidy.pred, select(Boston.test, medv))
tidy.res

rmse(tidy.res, truth=medv, estimate=.pred)
rsq(tidy.res, truth=medv, estimate=.pred)

metrics <- metric_set(rmse, rsq)
metrics(tidy.res, truth=medv, estimate=.pred)

tidy.res <- last_fit(tidy.mod, preprocessor=medv ~ crim + rm, split=split)
class(tidy.res)

tidy.res
tidy.res$.workflow[[1]]
tidy.res$.metrics[[1]]
tidy.res$.predictions[[1]]

extract_workflow(tidy.res)
collect_metrics(tidy.res)
collect_predictions(tidy.res)

## 워크플로우

tidy.mod <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")

tidy.wflow <- workflow() |> 
  add_model(tidy.mod)
class(tidy.wflow)
tidy.wflow

tidy.wflow <- workflow() |> 
  add_model(tidy.mod) |>  
  add_formula(medv ~ crim + rm)
tidy.wflow

tidy.fit <- fit(tidy.wflow, data=Boston.train)
class(tidy.fit)
tidy.fit

tidy.pred <- predict(tidy.fit, new_data=Boston.test)
tidy.pred

tidy.fit |> 
  update_formula(medv ~ crim + rm + age)

tidy.wflow |> 
  remove_formula() |> 
  add_variables(outcomes=medv, predictors=c(crim, rm, age))

tidy.res <- last_fit(tidy.wflow, split=split)
tidy.res

extract_workflow(tidy.res)
collect_metrics(tidy.res)
collect_predictions(tidy.res)

preproc <- list(m1=medv ~ crim,
                m2=medv ~ crim + rm,
                m3=medv ~ crim + rm + age)
tidy.mod <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")
models <- list(rpart=tidy.mod)
tidy.wflowset <- workflow_set(preproc=preproc, models=models)
class(tidy.wflowset)
tidy.wflowset

tidy.wflowset$info

tidy.wflowset$info[[2]]
tidy.wflowset$info[[2]]$workflow[[1]]

extract_workflow(tidy.wflowset, id="m2_rpart")

fit(tidy.wflowset$info[[2]]$workflow[[1]], data=Boston.train)

tidy.fits <- tidy.wflowset |> 
  mutate(fit=map(info, ~fit(.x$workflow[[1]], data=Boston.train)))
tidy.fits

tidy.fits$fit[[2]]
predict(tidy.fits$fit[[2]], new_data=Boston.test)

tidy.lastfits <- tidy.wflowset |> 
  mutate(lastfit=map(info, ~last_fit(.x$workflow[[1]], split=split)))
tidy.lastfits

tidy.lastfits$lastfit[[2]]
tidy.lastfits$lastfit[[2]]$.metrics[[1]]
tidy.lastfits$lastfit[[2]]$.predictions[[1]]

collect_metrics(tidy.lastfits$lastfit[[2]])
collect_predictions(tidy.lastfits$lastfit[[2]])

map(tidy.lastfits$lastfit, collect_metrics)
map(tidy.lastfits$lastfit, collect_predictions)

## 레시피

recipe <- recipe(medv ~ crim + rm, data=Boston.train) |> 
  step_normalize(all_numeric(), -all_outcomes())
class(recipe)
recipe

tidy.mod <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")

tidy.wflow <- workflow() |> 
  add_model(tidy.mod) |> 
  add_recipe(recipe)
tidy.wflow

tidy.fit <- fit(tidy.wflow, data=Boston.train)
tidy.fit

tidy.fit |> 
  extract_recipe() 

tidy.pred <- predict(tidy.fit, new_data=Boston.test)
tidy.pred

tidy.res <- bind_cols(tidy.pred, select(Boston.test, medv))
tidy.res
metrics <- metric_set(rmse, rsq)
metrics(tidy.res, truth=medv, estimate=.pred)

## 표본재추출 및 모델평가 

set.seed(123)
bs <- bootstraps(data=Boston.train, times=10)
class(bs)
bs
set.seed(123)
folds <- vfold_cv(data=Boston.train, v=10)
class(folds)
folds

tidy.mod <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")
tidy.wflow <- workflow() |> 
  add_model(tidy.mod) |> 
  add_formula(medv ~ crim + rm)
tidy.res <- fit_resamples(tidy.wflow, resamples=folds, 
                          control=control_resamples(save_pred=TRUE))
class(tidy.res)

fit_resamples(tidy.mod, preprocessor=medv ~ crim + rm, resamples=folds,
              control=control_resamples(save_pred=TRUE))

tidy.res
tidy.res$.metrics
tidy.res$.predictions

collect_metrics(tidy.res)
collect_predictions(tidy.res)

collect_metrics(tidy.res, summarize=FALSE)
collect_predictions(tidy.res, summarize=TRUE)

library(parallel)
detectCores(logical=FALSE)
detectCores(logical=TRUE)
detectCores()

library(future)
plan(multisession, workers=4)

availableCores()
availableWorkers()

fit_resamples(tidy.wflow, resamples=folds, 
              control=control_resamples(save_pred=TRUE))

plan(sequential)

preproc <- list(m1=medv ~ crim,
                m2=medv ~ crim + rm,
                m3=medv ~ crim + rm + age)
tidy.mod <- decision_tree() |> 
  set_engine("rpart") |> 
  set_mode("regression")
models <- list(rpart=tidy.mod)
tidy.wflowset <- workflow_set(preproc=preproc, models=models)
tidy.wflowset

tidy.fits <- workflow_map(tidy.wflowset, fn="fit_resamples",
                          resamples=folds, control=control_resamples(save_pred=TRUE),
                          verbose=TRUE, seed=123)
class(tidy.fits)

tidy.fits
tidy.fits$option
tidy.fits$result

tidy.fits$result[[2]]$.metrics

collect_metrics(tidy.fits)
collect_metrics(tidy.fits) |> 
  filter(.metric=="rmse")

# [그림 7-3]
windows(width=7.0, height=5.5)
library(ggrepel)
autoplot(tidy.fits, metric="rsq") +
  geom_text_repel(aes(label=wflow_id), nudge_x=0.2, nudge_y=0.05) +
  theme_bw() +
  theme(legend.position="none")

## 모델튜닝

get_from_env("decision_tree_args")

tidy.tune <- decision_tree(tree_depth=tune(),
                           min_n=tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")
tidy.tune

tidy.param <- extract_parameter_set_dials(tidy.tune)
class(tidy.param)
tidy.param

tidy.param |> 
  extract_parameter_dials("tree_depth")
tidy.param |> 
  extract_parameter_dials("min_n")

grid_regular(tidy.param, levels=3)

grid_regular(tidy.param, levels=c(tree_depth=3, min_n=3))

tree_depth()
min_n()
grid_regular(tree_depth(), min_n(), levels=3)

set.seed(123)
grid_random(tidy.param, size=9)

hyper.grid <- crossing(tree_depth=c(2, 10, 20), min_n=c(3, 15, 30))
hyper.grid

tidy.wflow <- workflow() |> 
  add_model(tidy.tune) |> 
  add_formula(medv ~ crim + rm)
tidy.wflow
tidy.res <- tune_grid(tidy.wflow, resamples=folds, 
                      control=control_resamples(save_pred=TRUE),
                      grid=hyper.grid)
class(tidy.res)
tidy.res

collect_metrics(tidy.res)

# [그림 7-4]
windows(width=7.0, height=5.5)
autoplot(tidy.res) +
  geom_line(linewidth=1.5, alpha=0.6) +
  geom_point(size=2) +
  scale_colour_viridis_d(option="plasma", begin=0.9, end=0) +
  facet_wrap(vars(.metric), scales="free", ncol=2) +
  theme_bw() +
  theme(legend.position="top")

show_best(tidy.res, metric="rmse")

tidy.best <- select_best(tidy.res, metric="rmse")
tidy.best

tidy.wflow.final <- tidy.wflow |> 
  finalize_workflow(tidy.best)
tidy.wflow.final

tidy.fit.final <- fit(tidy.wflow.final, data=Boston.train)
tidy.fit.final

tidy.res.final <- predict(tidy.fit.final, new_data=Boston.test) |> 
  bind_cols(select(Boston.test, medv))
tidy.res.final
metrics <- metric_set(rmse, rsq)
metrics(tidy.res.final, truth=medv, estimate=.pred)

tidy.res.final <- last_fit(tidy.wflow.final, split=split)
collect_metrics(tidy.res.final)

## 사례: 신용평가

library(tidymodels)
tidymodels_prefer()

library(modeldata)
str(credit_data)
levels(credit_data$Status)

set.seed(123)
split <- initial_split(data=credit_data, prop=0.7, strata=Status)
credit.train <- training(split)
credit.test <- testing(split)
dim(credit.train)
dim(credit.test)

set.seed(123)
folds <- vfold_cv(data=credit.train, v=10)
folds

# 페널티로지스틱회귀 

get_from_env("logistic_reg")

get_from_env("logistic_reg_args")

lr.tune <- logistic_reg(penalty=tune(), mixture=1) |> 
  set_engine("glmnet") 

library(naniar)
miss_var_summary(credit_data)

library(caret)
nearZeroVar(credit_data, saveMetrics=TRUE)

lr.recipe <- recipe(Status ~ ., data=credit.train) |>
  step_nzv(all_predictors()) |> 
  step_impute_knn(all_predictors(), neighbors=5) |>
  step_normalize(all_numeric(), -all_outcomes()) |> 
  step_dummy(all_nominal(), -all_outcomes())

lr.wflow <- workflow() |> 
  add_model(lr.tune) |> 
  add_recipe(lr.recipe)
lr.wflow

hyper.grid <- tibble(penalty=10^seq(from=-4, to=-1, length.out=30))
hyper.grid
slice_max(hyper.grid, order_by=penalty, n=5)
slice_min(hyper.grid, order_by=penalty, n=5)

lr.res <- tune_grid(lr.wflow, resamples=folds, 
                    control=control_resamples(save_pred=TRUE),
                    grid=hyper.grid,
                    metrics=metric_set(accuracy, roc_auc))
lr.res

collect_metrics(lr.res)

collect_metrics(lr.res) |> 
  filter(.metric=="roc_auc")

# [그림 7-5]
windows(width=7.0, height=5.5)
library(scales)
ggplot(data=filter(collect_metrics(lr.res), .metric=="roc_auc"),
       aes(x=penalty, y=mean)) +
  geom_line(linewidth=1.5, color="cornflowerblue", alpha=0.7) +
  geom_point(size=2, color="royalblue") +
  scale_x_log10(labels=label_number()) +
  labs(y="Area under the ROC Curve") +
  theme_bw() 

show_best(lr.res, metric="roc_auc", n=20) |> 
  arrange(penalty)
lr.best <- filter(collect_metrics(lr.res), .metric=="roc_auc") |> 
  arrange(penalty) |> 
  slice(15)
lr.best

lr.pred <- collect_predictions(lr.res, parameters=lr.best)
lr.pred

lr.auc <- roc_curve(lr.pred, Status, .pred_bad) |> 
  mutate(model="Logistic Regression")
lr.auc

# [그림 7-6]
windows(width=7.0, height=5.5)
autoplot(lr.auc)

# 랜덤포레스트

get_from_env("rand_forest")

get_from_env("rand_forest_args")

rf.tune <- rand_forest(mtry=tune(), min_n=tune(), trees=1000) |> 
  set_engine("ranger") |> 
  set_mode("classification")

rf.recipe <- recipe(Status ~ ., data=credit.train) |>
  step_nzv(all_predictors()) |> 
  step_impute_knn(all_predictors(), neighbors=5) 

rf.wflow <- workflow() |> 
  add_model(rf.tune) |> 
  add_recipe(rf.recipe)
rf.wflow

library(future)
plan(multisession)

set.seed(123)
rf.res <- tune_grid(rf.wflow, resamples=folds, 
                    control=control_resamples(save_pred=TRUE),
                    grid=25,
                    metrics=metric_set(accuracy, roc_auc))
rf.res

plan(sequential)

rf.tune <- rand_forest(mtry=tune(), min_n=tune(), trees=1000) |> 
  set_engine("ranger", num.threads=availableCores()) |> 
  set_mode("classification")

rf.recipe <- recipe(Status ~ ., data=credit.train) |>
  step_nzv(all_predictors()) |> 
  step_impute_knn(all_predictors(), neighbors=5) 
rf.wflow <- workflow() |> 
  add_model(rf.tune) |> 
  add_recipe(rf.recipe)
rf.wflow

set.seed(123)
rf.res <- tune_grid(rf.wflow, resamples=folds, 
                    control=control_resamples(save_pred=TRUE),
                    grid=25,
                    metrics=metric_set(accuracy, roc_auc))
rf.res

collect_metrics(rf.res)

show_best(rf.res, metric="roc_auc")

rf.best <- select_best(rf.res, metric="roc_auc")
rf.best

rf.pred <- collect_predictions(rf.res, parameters=rf.best)
rf.pred
rf.auc <- roc_curve(rf.pred, Status, .pred_bad) |> 
  mutate(model="Random Forest")
rf.auc

# [그림 7-7]
windows(width=7.0, height=5.5)
autoplot(rf.auc)

# 최종모델 

# [그림 7-8]
windows(width=7.0, height=5.5)
ggplot(data=bind_rows(rf.auc, lr.auc), 
       aes(x=1-specificity, y=sensitivity, col=model)) + 
  geom_path(lwd=1.5, alpha=0.8) +
  geom_abline(lty=3) + 
  coord_equal() + 
  scale_color_viridis_d(option="plasma", begin=0, end=0.5) +
  theme_bw() +
  theme(legend.position="top")

rf.mod.final <- 
  rand_forest(mtry=1, min_n=24, trees=1000) |> 
  set_engine("ranger", num.threads=availableCores(), importance="impurity") |> 
  set_mode("classification")

rf.wflow.final <- rf.wflow |> 
  update_model(rf.mod.final)
rf.wflow.final

set.seed(123)
rf.res.final <- last_fit(rf.wflow.final, split=split, 
                         metrics=metric_set(accuracy, roc_auc))
rf.res.final

collect_metrics(rf.res.final)

# [그림 7-9]
windows(width=7.0, height=5.5)
library(vip)
rf.res.final |> 
  extract_fit_parsnip() |> 
  vip(aesthetics=list(color="dimgray", fill="skyblue"))

# [그림 7-10]
windows(width=7.0, height=5.5)
rf.res.final  |>  
  collect_predictions() |> 
  roc_curve(Status, .pred_bad) |> 
  ggplot(aes(x=1-specificity, y=sensitivity)) +
  geom_path(linewidth=1.5, col="cornflowerblue") +
  geom_abline(linewidth=1, lty=3, col="darkred") +
  coord_equal() +
  theme_bw()

rf.res.final  |>  
  collect_predictions() |> 
  roc_curve(Status, .pred_bad) |> 
  autoplot()

###############
## 7.2 Caret ##
###############

## 데이터전처리

library(mlbench)
data(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)

library(caret)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes2$diabetes, p=0.7, list=FALSE)
pima.train <- PimaIndiansDiabetes2[index,]
pima.test <- PimaIndiansDiabetes2[-index,]
dim(pima.train)
dim(pima.test)
prop.table(table(pima.train$diabetes))
prop.table(table(pima.test$diabetes))
prop.table(table(PimaIndiansDiabetes2$diabetes))

library(skimr)
skim(pima.train)

prepro.missing <- preProcess(pima.train, method="knnImpute") 
class(prepro.missing)
prepro.missing

install.packages("RANN")
pima.train <- predict(prepro.missing, newdata=pima.train)
anyNA(pima.train)

prepro.range <- preProcess(pima.train, method="range")
prepro.range

pima.train <- predict(prepro.range, newdata=pima.train)
sapply(pima.train[c(1:8)], function(x) range(x))

pima.test <- predict(prepro.missing, newdata=pima.test)
pima.test <- predict(prepro.range, newdata=pima.test)
anyNA(pima.test)
sapply(pima.test[c(1:8)], function(x) range(x))

## 모델생성 및 모델튜닝

library(caret)
names(getModelInfo())

modelLookup("C5.0")

set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, method="C5.0")
class(caret.cv)

caret.cv

getTrainPerf(caret.cv)

caret.cv$results
library(dplyr)
slice_max(caret.cv$results, order_by=Accuracy)

caret.cv$resample
mean(caret.cv$resample$Accuracy)

caret.cv$bestTune

predict(caret.cv, newdata=pima.test, type="raw")
predict(caret.cv, newdata=pima.test)
predict(caret.cv, newdata=pima.test, type="prob")

caret.pred <- predict(caret.cv, newdata=pima.test)
table(pima.test$diabetes, caret.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==caret.pred)

confusionMatrix(data=caret.pred, reference=pima.test$diabetes, positive="pos", 
                mode="everything")

postResample(pred=caret.pred, obs=pima.test$diabetes)

control <- trainControl(method="cv", number=10, classProbs=TRUE, savePredictions="final", 
                        summaryFunction=twoClassSummary, selectionFunction="oneSE")

set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, 
                  method="C5.0",
                  metric="ROC",
                  trControl=control,
                  tuneLength=5)
caret.cv

control <- trainControl(method="cv", number=10, classProbs=TRUE, savePredictions="final",
                        summaryFunction=twoClassSummary, selectionFunction="oneSE",
                        search="random")
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, 
                  method="C5.0",
                  metric="ROC",
                  trControl=control,
                  tuneLength=5)
caret.cv

hyper.grid <- expand.grid(trials=c(1, 5, 10, 15, 20, 25, 30, 35, 40, 45),
                          model=c("tree", "rules"),
                          winnow=c(TRUE, FALSE))
hyper.grid

control <- trainControl(method="cv", number=10, classProbs=TRUE, savePredictions="final", 
                        summaryFunction=defaultSummary, selectionFunction="tolerance")
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, 
                  method="C5.0",
                  metric="Accuracy",
                  trControl=control,
                  tuneGrid=hyper.grid)
caret.cv

library(dplyr)
slice_max(caret.cv$results, order_by=Accuracy, n=5, with_ties=FALSE)

caret.cv$bestTune
mean(caret.cv$resample$Accuracy)

# [그림 7-12]
windows(width=7.0, height=5.5)
ggplot(caret.cv) +
  theme_bw() +
  theme(strip.background=element_rect(fill="lavender"),
        legend.position="top")

plot(caret.cv)

imp <- varImp(caret.cv)
class(imp)
imp

varImp(caret.cv, scale="FALSE")

# [그림 7-13]
windows(width=7.0, height=4.0)
ggplot(imp, top=5) +
  geom_bar(stat="identity", color="dimgray", fill="mistyrose") +
  theme_light()

plot(imp)

library(parallel)
library(doParallel)
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)
set.seed(123)
train(diabetes ~ ., data=pima.train, 
      method="C5.0",
      metric="Accuracy",
      trControl=trainControl(method="cv", number=10, classProbs=TRUE, savePredictions="final", 
                             summaryFunction=defaultSummary, selectionFunction="tolerance"),
      tuneGrid=expand.grid(trials=c(1, 5, 10, 15, 20, 25, 30, 35, 40, 45),
                           model=c("tree", "rules"),
                           winnow=c(TRUE, FALSE)))

stopCluster(cl)
registerDoSEQ() 

set.seed(123)
caret.fit <- train(diabetes ~ ., data=pima.train, 
                   method="C5.0",
                   metric="ROC",
                   trControl=trainControl(method="none", classProbs=TRUE),
                   tuneGrid=data.frame(trials=5, model="tree", winnow=TRUE))
caret.fit

caret.pred <- predict(caret.fit, newdata=pima.test)
head(caret.pred)
table(pima.test$diabetes, caret.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==caret.pred)

## 모델비교

library(caret)
control <- trainControl(method="cv", number=10, classProbs=TRUE)
set.seed(123)
caret.C5.0 <- train(diabetes ~ ., data=pima.train, 
                    method="C5.0",
                    metric="Accuracy",
                    trControl=control,
                    tuneLength=5)
set.seed(123)
caret.svm <- train(diabetes ~ ., data=pima.train, 
                   method="svmRadial",
                   metric="Accuracy",
                   trControl=control,
                   tuneLength=5)
set.seed(123)
caret.rf <- train(diabetes ~ ., data=pima.train, 
                  method="rf",
                  metric="Accuracy",
                  trControl=control,
                  tuneLength=5)

caret.compare <- resamples(list(C5.0=caret.C5.0, SVM=caret.svm, RF=caret.rf))
class(caret.compare)

caret.compare$values
summary(caret.compare)

# [그림 7-14]
windows(width=8.0, height=4.5)
bwplot(caret.compare, 
       strip=strip.custom(par.strip.text=list(cex=0.7), bg="wheat"),
       scales=list(x=list(relation="free"), y=list(relation="free")))

caret.diff <- diff(caret.compare)
summary(caret.diff)

# [그림 7-15]
windows(width=8.0, height=4.5)
bwplot(caret.diff, 
       strip=strip.custom(par.strip.text=list(cex=0.7), bg="beige"),
       scales=list(x=list(relation="free"), y=list(relation="free")))

## 앙상블

library(ipred)
set.seed(123)
bag.fit <- bagging(diabetes ~ ., data=pima.train, nbagg=25)
class(bag.fit)

bag.pred <- predict(bag.fit, newdata=pima.test)
head(bag.pred)
table(pima.test$diabetes, bag.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==bag.pred)

control <- trainControl(method="cv", number=10)
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, 
                  method="treebag",
                  trControl=control)
caret.cv

library(adabag)
set.seed(123)
boost.fit <- boosting(diabetes ~ ., data=pima.train)
class(boost.fit)

boost.pred <- predict(boost.fit, newdata=pima.test)
head(boost.pred$class)

boost.pred$confusion
1 - boost.pred$error

library(irr)
kappa2(data.frame(pima.test$diabetes, boost.pred$class))$value
library(vcd)
Kappa(boost.pred$confusion)

set.seed(123)
boost.cv <- boosting.cv(diabetes ~ ., data=pima.train)

boost.cv$confusion
1 - boost.cv$error

control <- trainControl(method="cv", number=5)
set.seed(123)
caret.cv <- train(diabetes ~ ., data=pima.train, 
                  method="AdaBoost.M1",
                  trControl=control)
caret.cv
getTrainPerf(caret.cv)

caret.pred <- predict(caret.cv, newdata=pima.test)
head(caret.pred)
table(pima.test$diabetes, caret.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==caret.pred)

## 레시피

library(caret)
set.seed(123)
index <- createDataPartition(y=PimaIndiansDiabetes2$diabetes, p=0.7, list=FALSE)
pima.train <- PimaIndiansDiabetes2[index,]
pima.test <- PimaIndiansDiabetes2[-index,]

library(recipes)
recipe.step <- recipe(diabetes ~ ., data=pima.train) |> 
  step_impute_knn(all_predictors(), neighbors=5) |> 
  step_range(all_numeric(), -all_outcomes()) 
recipe.step

set.seed(123)
caret.cv <- train(recipe.step, data=pima.train, 
                  method="C5.0",
                  metric="Accuracy",
                  trControl=trainControl(method="cv", number=10),
                  tuneLength=5)
caret.cv
caret.cv$bestTune
getTrainPerf(caret.cv)

caret.cv$recipe
predictors(caret.cv)

caret.pred <- predict(caret.cv, newdata=pima.test)
head(caret.pred)
table(pima.test$diabetes, caret.pred, dnn=c("Actual", "Predicted"))
mean(pima.test$diabetes==caret.pred)

## 사례: 주택가격

library(caret)
data(Sacramento)
str(Sacramento)

set.seed(123)
index <- createDataPartition(y=Sacramento$price, p=0.7, list=FALSE)
Sacramento.train <- Sacramento[index,]
Sacramento.test <- Sacramento[-index,]
dim(Sacramento.train)
dim(Sacramento.test)

# [그림 7-16]
windows(width=8.0, height=4.5)
p1 <- ggplot(Sacramento.train, aes(x=price)) +
  geom_histogram(color="darkblue", fill="cornflowerblue")  +
  labs(title="Histogram of price") +
  theme_minimal()
p2 <- ggplot(Sacramento.train, aes(x=log(price))) +
  geom_histogram(color="darkred", fill="salmon")  +
  labs(title="Histogram of log price") +
  theme_minimal()
library(patchwork)
p1 + p2

Sacramento.train$price <- log(Sacramento.train$price)
Sacramento.test$price <- log(Sacramento.test$price)

prepro.dummy <- dummyVars(~ ., data=Sacramento.train, fullRank=FALSE)

Sacramento.train <- as.data.frame(predict(prepro.dummy, newdata=Sacramento.train))
Sacramento.test <- as.data.frame(predict(prepro.dummy, newdata=Sacramento.test))

# 페널티회귀

control <- trainControl(method="repeatedcv", number=5, repeats=5, savePredictions="final")

set.seed(123)
caret.glmnet <- train(price ~ ., data=Sacramento.train,
                      method="glmnet",
                      preProcess=c("zv", "center", "scale"),
                      trControl=control,
                      tuneLength=10)
caret.glmnet

modelLookup("glmnet")
caret.glmnet$results

caret.glmnet$bestTune
getTrainPerf(caret.glmnet)

# 랜덤포레스트

library(parallel)
library(doParallel)
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)

set.seed(123)
control <- trainControl(method="repeatedcv", number=5, repeats=5, savePredictions="final")
caret.rf <- train(price ~ ., data=Sacramento.train,
                  method="ranger",
                  trControl=control,
                  tuneLength=10)
caret.rf

stopCluster(cl)
registerDoSEQ()

modelLookup("ranger")
caret.rf$results

caret.rf$bestTune
getTrainPerf(caret.rf)

# 최종모델 

caret.compare <- resamples(list(Glmnet=caret.glmnet, RF=caret.rf))
summary(caret.compare)

# [그림 7-17]
windows(width=8.0, height=4.5)
bwplot(caret.compare, 
       strip=strip.custom(par.strip.text=list(cex=0.7), bg="lavender"),
       scales=list(x=list(relation="free"), y=list(relation="free")))

caret.pred <- exp(predict(caret.rf, newdata=Sacramento.test))
head(caret.pred)

options(scipen=999)
postResample(pred=caret.pred, obs=exp(Sacramento.test$price))
options(scipen=0)
