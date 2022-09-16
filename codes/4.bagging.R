## building decision trees models
if(T){
  
  rm(list=ls())
  library(pacman)
  p_load(dplyr,
         grid,
         foreach,
         doParallel,
         ipred, # for fitting bagged decision trees
         gridExtra,
         ggplot2,
         caret,# meta engine for decision tree application
         magrittr,
         ranger,# a c++ implementation of random forest
         vip, # for important variables
         rpart,       # direct engine for decision tree application
         # Model interpretability packages
         rpart.plot,  # for plotting decision trees
         vip,         # for feature importance
         pdp)         # for feature effects
  
  setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
  load("./data/train_test.Rdata")
  
  ## filter data for one dataset: train data
  x.train = data.list[[1]]
  x.test = data.list[[2]]
  
  # make bootstrapping reproducible
  set.seed(123)
  
  # train bagged model
  bag1 <- bagging(
    formula = Dx ~ .,
    data = x.train,
    nbagg = 100,  
    coob = TRUE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  bag1
  
  bag2 <- train(
    Dx ~ .,
    data = x.train,
    method = "treebag",
    trControl = trainControl(method = "cv", number = 10),
    nbagg = 200,  
    control = rpart.control(minsplit = 2, cp = 0)
  )
  
  bag2
  
  ## parallelize
  # Create a parallel socket cluster
  cl <- makeCluster(8) # use 8 workers
  registerDoParallel(cl) # register the parallel backend
  
  # Fit trees in parallel and compute predictions on the test set
  predictions <- foreach(
    icount(200), 
    .packages = "rpart", 
    .combine = cbind
  ) %dopar% {
    # bootstrap copy of training data
    index <- sample(nrow(x.train), replace = TRUE)
    train_boot <- x.train[index, ]  
    
    # fit tree to bootstrap copy
    bagged_tree <- rpart(
      Dx ~ ., 
      control = rpart.control(minsplit = 2, cp = 0),
      data = train_boot
    ) 
    
    predict(bagged_tree, newdata = x.test)
  }
  
  predictions[1:5, 1:7]
  
  ##We can then do some data wrangling to compute and plot the RMSE 
  predictions %>%
    as.data.frame() %>%
    mutate(
      observation = 1:n(),
      actual = x.test$Dx) %>%
    tidyr::gather(tree, predicted, -c(observation, actual)) %>%
    group_by(observation) %>%
    mutate(tree = stringr::str_extract(tree, '\\d+') %>% as.numeric()) %>%
    ungroup() %>%
    arrange(observation, tree) %>%
    group_by(observation) %>%
    mutate(avg_prediction = cummean(predicted)) %>%
    group_by(tree) %>%
    summarize(RMSE = RMSE(avg_prediction, actual)) %>%
    ggplot(aes(tree, RMSE)) +
    geom_line() +
    xlab('Number of trees')
  
  # Shutdown parallel cluster
  stopCluster(cl)
 ## feature importance
 vip(dt3, num_features = 40, bar = F)
  vi <- varImp(dt3)
  write.csv(vi,file="./results/tables/decisionTree_varImp.csv")

  ## save data for plotting
  saveRDS(model_list,"./data/knn_model.Rdata")
  
 
}

