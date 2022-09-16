## here we train multivariate adaptive regression splines (MARS)

if(T){
  
  rm(list=ls())
  pacman::p_load(dplyr,
                 grid,
                 earth,
                 caret,
                 magrittr,
                 vip, # for important variables
                 pdp,
                 gridExtra,
                 ggplot2)
  
  setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
  load("./data/train_test.Rdata")
  
  ## filter data for one dataset: train data
  x.train = data.list[[1]]
  x.test = data.list[[2]]
  
  # create a tuning grid
  hyper_grid <- expand.grid(
    degree = 1:3, 
    nprune = seq(2, 100, length.out = 10) %>% floor()
  )
  
  # Cross-validated model
  set.seed(123)  # for reproducibility
  cv_mars <- train(
    x = subset(x.train, select = -Dx),
    y = x.tra$Dx,
    method = "earth",
    metric = "RMSE",
    trControl = trainControl(method = "cv", number = 10),
    tuneGrid = hyper_grid
  )
  
  # View results
  cv_mars$bestTune
  
  cv_mars$results %>%
    filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)
  
  # Plot 
  p = ggplot(cv_mars)
  
  # variable importance plots
  p1 <- vip(cv_mars, num_features = 40, geom = "point", value = "gcv") + ggtitle("GCV")
  p2 <- vip(cv_mars, num_features = 40, geom = "point", value = "rss") + ggtitle("RSS")
  
  gg = gridExtra::grid.arrange(p1, p2, ncol = 2)
  
  # extract coefficients, convert to tidy data frame, and
  # filter for interaction terms
  cv_mars$finalModel %>%
    coef() %>%  
    broom::tidy() %>%  
    filter(stringr::str_detect(names, "\\*")) 
  