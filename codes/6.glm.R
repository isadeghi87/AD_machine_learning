## building random forest models
if(T){
  
  rm(list=ls())
  pacman::p_load(dplyr,
         grid,
         caret,
         recipes, # for feature engineering
         glmnet,
         magrittr,
         h2o,# a java-based implementation of random forest
         vip, # for important variables
         gridExtra,
         ggplot2)
  
  setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
  load("./data/train_test.Rdata")
  
  ## filter data for one dataset: train data
  x.train = data.list[[1]]
  x.test = data.list[[2]]
  
  # Create training  feature matrices
  # we use model.matrix(...)[, -1] to discard the intercept
  X <- model.matrix(Dx ~ ., x.train)[, -1]
  Y <- x.train$Dx
  
    ##Recall that  
  # λ is a tuning parameter that helps to control our model from 
  # over-fitting to the training data. To identify the optimal  
  # λ  value we can use k-fold cross-validation (CV). 
 
  
  ctrl <- trainControl(method = "cv",
                        number = 5,
                        # summaryFunction = twoClassSummary,
                        verboseIter = TRUE,
                        allowParallel = T,
                        savePredictions = TRUE,
                        classProbs  = TRUE)
  
  # grid search across 
  glm_model <- train(Dx ~ .,
                     x.train,
                     method = "glmnet",
                     tuneGrid = expand.grid(
                       alpha = 0:1,
                       lambda = 0:10/10),
                     trControl = ctrl)
  
  # model with lowest RMSE
  glm_model$bestTune
  ##   alpha     lambda
  ## 7   0.1 0.02007035
  
  # results for model with lowest RMSE
 glm_model$results %>%
    filter(alpha == glm_model$bestTune$alpha, lambda == glm_model$bestTune$lambda)

  
  # plot cross-validated RMSE
  ggplot(glm_model)
  
  # predict  on training data
  pred <- predict(glm_model, X)
  
  # compute RMSE of transformed predicted
  RMSE(pred, Y)
  
  ## top important vars
  topvar = vip::vi(glm_model)
  
  # variable importance plot
  vip::vip(glm_model)
}

