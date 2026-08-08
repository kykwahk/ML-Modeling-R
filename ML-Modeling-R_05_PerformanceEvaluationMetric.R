
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

#########################
## 제5장 성능평가 지표 ##
#########################

##################
## 5.1 회귀모델 ##
##################

## 사례

library(yardstick)
head(solubility_test)
str(solubility_test)

rmse(data=solubility_test, truth=solubility, estimate=prediction)

rmse(data=solubility_test, truth=solubility, estimate=prediction)$.estimate^2

mae(data=solubility_test, truth=solubility, estimate=prediction)
rsq(data=solubility_test, truth=solubility, estimate=prediction)

rmse_vec(truth=solubility_test$solubility, estimate=solubility_test$prediction)

mae_vec(truth=solubility_test$solubility, estimate=solubility_test$prediction)
rsq_vec(truth=solubility_test$solubility, estimate=solubility_test$prediction)

metrics(data=solubility_test, truth=solubility, estimate=prediction)

multimetrics <- metric_set(rmse, rsq)
multimetrics(data=solubility_test, truth=solubility, estimate=prediction)

##################
## 5.2 분류모델 ##
##################

## 사례: 이진분류

library(yardstick)
head(two_class_example)
str(two_class_example)

mn_log_loss(data=two_class_example, truth=truth, Class1)

mn_log_loss_vec(truth=two_class_example$truth, estimate=two_class_example$Class1)

conf_mat(data=two_class_example, truth=truth, estimate=predicted)

accuracy(data=two_class_example, truth=truth, estimate=predicted)
sens(data=two_class_example, truth=truth, estimate=predicted)
spec(data=two_class_example, truth=truth, estimate=predicted)
precision(data=two_class_example, truth=truth, estimate=predicted)
recall(data=two_class_example, truth=truth, estimate=predicted)

accuracy_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)
sens_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)
spec_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)
precision_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)
recall_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)

f_meas(data=two_class_example, truth=truth, estimate=predicted)
f_meas_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)
kap(data=two_class_example, truth=truth, estimate=predicted)
kap_vec(truth=two_class_example$truth, estimate=two_class_example$predicted)

roc_curve(data=two_class_example, truth=truth, Class1)

# [그림 5-16]
windows(width=7.0, height=5.5)
library(ggplot2)
ggplot(roc_curve(data=two_class_example, truth=truth, Class1),
       aes(x=1-specificity, y=sensitivity)) +
  geom_path(linewidth=2, col="cornflowerblue") +
  geom_abline(linewidth=1, lty=3, col="red") +
  coord_equal() +
  theme_bw()

autoplot(roc_curve(data=two_class_example, truth=truth, Class1))

roc_auc(data=two_class_example, truth=truth, Class1)
roc_auc_vec(truth=two_class_example$truth, estimate=two_class_example$Class1)

# [그림 5-17]
windows(width=7.0, height=5.5)
library(Epi)
ROC(test=two_class_example$Class1, 
    stat=relevel(two_class_example$truth, ref="Class2"), lwd=3)

metrics(data=two_class_example, truth=truth, estimate=predicted)

metrics(data=two_class_example, truth=truth, estimate=predicted, Class1)

multimetrics <- metric_set(sens, spec, f_meas)
multimetrics(data=two_class_example, truth=truth, estimate=predicted)

## 사례: 다중분류

library(yardstick)
head(hpc_cv)
str(hpc_cv)

library(dplyr)
hpc_cv.fold1 <- filter(hpc_cv, Resample=="Fold01")
mn_log_loss(data=hpc_cv.fold1, truth=obs, VF:L)

mn_log_loss_vec(truth=hpc_cv.fold1$obs,
                estimate=matrix(c(hpc_cv.fold1$VF, hpc_cv.fold1$F, hpc_cv.fold1$M, hpc_cv.fold1$L),
                                ncol=4))

hpc_cv |> 
  group_by(Resample) |> 
  mn_log_loss(truth=obs, VF:L)

conf_mat(data=hpc_cv.fold1, truth=obs, estimate=pred)

accuracy(data=hpc_cv.fold1, truth=obs, estimate=pred)
sens(data=hpc_cv.fold1, truth=obs, estimate=pred)
spec(data=hpc_cv.fold1, truth=obs, estimate=pred)
precision(data=hpc_cv.fold1, truth=obs, estimate=pred)
recall(data=hpc_cv.fold1, truth=obs, estimate=pred)
f_meas(data=hpc_cv.fold1, truth=obs, estimate=pred)
kap(data=hpc_cv.fold1, truth=obs, estimate=pred)

accuracy_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)
sens_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)
spec_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)
precision_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)
recall_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)
f_meas_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)
kap_vec(truth=hpc_cv.fold1$obs, estimate=hpc_cv.fold1$pred)

f_meas(data=hpc_cv.fold1, truth=obs, estimate=pred, estimator="macro_weighted")
f_meas(data=hpc_cv.fold1, truth=obs, estimate=pred, estimator="micro")

hpc_cv |> 
  group_by(Resample) |> 
  accuracy(obs, pred)

roc_curve(data=hpc_cv.fold1, truth=obs, VF:L)

# [그림 5-18]
windows(width=7.0, height=7.0)
roc_curve(data=hpc_cv.fold1, truth=obs, VF:L) |>
  autoplot()

roc_auc(data=hpc_cv.fold1, truth=obs, VF:L)
roc_auc_vec(truth=hpc_cv.fold1$obs,
            estimate=matrix(c(hpc_cv.fold1$VF, hpc_cv.fold1$F, hpc_cv.fold1$M, hpc_cv.fold1$L),
                            ncol=4))

roc_auc(data=hpc_cv.fold1, truth=obs, VF:L, estimator="macro")
roc_auc(data=hpc_cv.fold1, truth=obs, VF:L, estimator="macro_weighted")

metrics(data=hpc_cv.fold1, truth=obs, estimate=pred)
metrics(data=hpc_cv.fold1, truth=obs, estimate=pred, VF:L)

multimetrics <- metric_set(sens, spec, f_meas)
multimetrics(data=hpc_cv.fold1, truth=obs, estimate=pred)

hpc_cv |> 
  group_by(Resample) |> 
  metrics(truth=obs, estimate=pred)
multimetrics <- metric_set(sens, spec, f_meas)
hpc_cv |> 
  group_by(Resample) |> 
  multimetrics(truth=obs, estimate=pred)

# [그림 5-19]
windows(width=7.0, height=7.0)
hpc_cv |>
  group_by(Resample) |>
  roc_curve(truth=obs, VF:L) |>
  autoplot()

hpc_cv |>
  group_by(Resample) |>
  roc_auc(truth=obs, VF:L)
