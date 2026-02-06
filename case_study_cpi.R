library(vimp)
library(cpi)
library(mlr3)
library(mlr3learners)
library(methods)
library(glmnet)
library(randomForest) 
library(rpart)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data_name = 'rosmap86'
train = read.csv(paste0(data_name,'_train.csv'))
tar = "y_reg"

alpha=0.1
p = ncol(train) - 1
ml_model = lrn(
  "regr.ranger",
  num.trees = 200,                 # n_estimators
  mtry = floor(p / 3),             # max_features = 1/3
  min.node.size = 5,               # min_samples_leaf
  replace = FALSE                  # bootstrap = FALSE
)

task = as_task_regr(x = train, target = tar) 
cpi_lm_log = cpi(task =  as_task_regr(train, target = tar), learner = ml_model, alpha = 0.1,
                 resampling = rsmp("holdout"),
                 test = "t", measure = 'regr.mse', log = FALSE)
cpi_lm_log$lb = cpi_lm_log$CPI-qnorm(1-alpha/2)*cpi_lm_log$SE
cpi_lm_log$ub = cpi_lm_log$CPI+qnorm(1-alpha/2)*cpi_lm_log$SE
cpi_new = cpi_lm_log[, c("Variable", "CPI", "estimate", "p.value","lb","ub")]
names(cpi_new) <- c("feature", "cpi", "estimate", "pval","lb","ub")
cpi = cpi_new[order(cpi_new$pval), ]
cpi
write.csv(cpi,paste0('Output_Data/Case_Study/',data_name,'_CPI_RandomForest.csv'))
