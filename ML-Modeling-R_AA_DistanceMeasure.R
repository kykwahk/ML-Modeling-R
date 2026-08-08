
####################################################
## R을 이용한 머신러닝: 모델링, Tidymodels, Caret ##
## (곽기영, 도서출판 청람)                        ## 
####################################################

######################
## 부록 A. 거리측정 ##
######################

##############
## A.2 사례 ##
##############

library(MASS)
str(survey)

library(dplyr)
survey.num <- select(survey, where(is.numeric))
str(survey.num)

diss.num <- dist(x=survey.num, method="euclidean")
as.matrix(diss.num)[1:5, 1:5]

library(proxy)
similarity <- simil(x=survey.num, method="cosine")
as.matrix(similarity)[1:5, 1:5]
1-as.matrix(similarity)[1:5, 1:5]
distance <- proxy::dist(x=survey.num, method="cosine")
as.matrix(distance)[1:5, 1:5]

survey.cat <- select(survey, where(is.factor))
str(survey.cat)

library(fastDummies)
survey.onehot <- dummy_cols(survey.cat,
                            remove_first_dummy=FALSE, 
                            remove_selected_columns=TRUE,
                            ignore_na=TRUE)
str(survey.onehot)

diss.cat <- stats::dist(x=survey.onehot, method="manhattan")
as.matrix(diss.cat)[1:5, 1:5]
diss.cat <- stats::dist(x=survey.onehot, method="binary")
as.matrix(diss.cat)[1:5, 1:5]

library(cluster)
diss.mixed <- daisy(x=survey, metric="gower")
as.matrix(diss.mixed)[1:5, 1:5]
