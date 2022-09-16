## building k-nearest neigbours models
if(T){
  
  rm(list=ls())
  library(pacman)
  p_load(dplyr,grid,
         caret,randomForest,magrittr,
         ranger,# a c++ implementation of random forest
         h2o,# a java-based implementation of random forest
         vip, # for important variables
         gridExtra,ggplot2)
  
  setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
  load("./data/train_test.Rdata")
  
  ## filter data for one dataset: train data
  x.train = data.list[[1]]
  x.test = data.list[[2]]
  
  set.seed(42)
  myGrid <- expand.grid(mtry = c(2, 10, 20, 50, 90)
                        # splitrule = c("gini", "extratrees"),
                        # min.node.size = 1
  ) ## Minimal node size; default 1 for classification
  
  # Perform crossvalidation
  ctrl1 <- trainControl(method = "repeatedcv",
                        number = 10,
                        repeats = 5,
                        summaryFunction = twoClassSummary,
                        verboseIter = TRUE,
                        allowParallel = T,
                        savePredictions = TRUE,
                        classProbs  = TRUE)
  
  # Create a hyperparameter grid search
  hyper_grid <- expand.grid(k = seq(3, 25, by = 2))
  
  # Execute grid search
  knn_model <- train(
    Dx ~ .,
    x.train,
    method = "knn",
    tuneGrid = hyper_grid,
    preProc = c("center", "scale"),
    # metric="ROC",
    trControl = ctrl1)
  
 knn_p = ggplot(knn_model)+
    theme_bw()+
   labs(title = "Knn model")
  
  ggsave("./results/figures/knn_model.pdf",plot = knn_p,
         width = 8,height = 5)
  
  # Create confusion matrix
  cm <- confusionMatrix(knn_model$pred$pred, knn_model$pred$obs)
  cm$byClass[c(1:2, 11)]  # sensitivity, specificity, & accuracy
  
  ## feature importance
  vi <- varImp(knn_model)
  write.csv(vi,file="./results/tables/knn_varImp.csv")

  ## save data for plotting
  saveRDS(model_list,"./data/knn_model.Rdata")
  
 
}

