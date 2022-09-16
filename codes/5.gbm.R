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
         gbm,
         xgboost,
         h2o,
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
  
  h2o.init(max_mem_size = "5g")
  
  train_h2o <- as.h2o(x.train)
  response <- "Dx"
  predictors <- setdiff(colnames(x.train), response)
  
  # run a basic GBM model
  set.seed(123)  # for reproducibility
  gbm1 <- gbm(
    formula = Dx ~ .,
    data = x.train,
    distribution = "gaussian",  # SSE loss function
    n.trees = 5000,
    shrinkage = 0.1,
    interaction.depth = 3,
    n.minobsinnode = 10,
    cv.folds = 10
  )
  
  # find index for number trees with minimum CV error
  best <- which.min(gbm1$cv.error)
  
  # get MSE and compute RMSE
  sqrt(gbm1$cv.error[best])
 
  # plot error curve
  gbm.perf(gbm1, method = "cv")
 
  # create grid search
  hyper_grid <- expand.grid(
    learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
    RMSE = NA,
    trees = NA,
    time = NA
  )
  
  # execute grid search
  for(i in seq_len(nrow(hyper_grid))) {
    
    # fit gbm
    set.seed(123)  # for reproducibility
    train_time <- system.time({
      m <- gbm(
        formula = Dx ~ .,
        data = x.train,
        distribution = "gaussian",
        n.trees = 5000, 
        shrinkage = hyper_grid$learning_rate[i], 
        interaction.depth = 3, 
        n.minobsinnode = 10,
        cv.folds = 10 
      )
    })
    
    # add SSE, trees, and training time to results
    hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
    hyper_grid$trees[i] <- which.min(m$cv.error)
    hyper_grid$Time[i]  <- train_time[["elapsed"]]
    
  }
  
  # results
  arrange(hyper_grid, RMSE)
  
  #Next, we'll set our learning rate at the optimal learning rate and tune
  #the tree specific hyperparameters (interaction.depth and n.minobsinnode).
  
  # search grid
  hyper_grid <- expand.grid(
    n.trees = 6000,
    shrinkage = 0.01,
    interaction.depth = c(3, 5, 7),
    n.minobsinnode = c(5, 10, 15)
  )
  
  # create model fit function
  model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
    set.seed(123)
    m <- gbm(
      formula = Dx ~ .,
      data = x.train,
      distribution = "gaussian",
      n.trees = n.trees,
      shrinkage = shrinkage,
      interaction.depth = interaction.depth,
      n.minobsinnode = n.minobsinnode,
      cv.folds = 10
    )
    # compute RMSE
    sqrt(min(m$cv.error))
  }
  
  # perform search grid with functional programming
  hyper_grid$rmse <- purrr::pmap_dbl(
    hyper_grid,
    ~ model_fit(
      n.trees = ..1,
      shrinkage = ..2,
      interaction.depth = ..3,
      n.minobsinnode = ..4
    )
  )
  
  # results
  arrange(hyper_grid, rmse)
  
  # stochastic GBM ####
  #The following uses h2o to implement a stochastic GBM. 
  # We use the optimal hyperparameters found in the 
  # previous section and build onto this by assessing a 
  # range of values for subsampling rows and columns before
  # each tree is built, 
 
  # refined hyperparameter grid
  hyper_grid <- list(
    sample_rate = c(0.5, 0.75, 1),              # row subsampling
    col_sample_rate = c(0.5, 0.75, 1),          # col subsampling for each split
    col_sample_rate_per_tree = c(0.5, 0.75, 1)  # col subsampling for each tree
  )
  
  # random grid search strategy
  search_criteria <- list(
    strategy = "RandomDiscrete",
    stopping_metric = "mse",
    stopping_tolerance = 0.001,   
    stopping_rounds = 10,         
    max_runtime_secs = 60*60      
  )
  
  # perform grid search 
  grid <- h2o.grid(
    algorithm = "gbm",
    grid_id = "gbm_grid",
    x = predictors, 
    y = response,
    training_frame = train_h2o,
    hyper_params = hyper_grid,
    ntrees = 6000,
    learn_rate = 0.01,
    max_depth = 7,
    min_rows = 5,
    nfolds = 10,
    stopping_rounds = 10,
    stopping_tolerance = 0,
    search_criteria = search_criteria,
    seed = 123
  )
  
  # collect the results and sort by our model performance metric of choice
  grid_perf <- h2o.getGrid(
    grid_id = "gbm_grid", 
    sort_by = "mse", 
    decreasing = FALSE
  )
  
  grid_perf
  
  #Grab the model_id for the top model, chosen by cross validation error
  best_model_id <- grid_perf@model_ids[[1]]
  best_model <- h2o.getModel(best_model_id)
  
  # Now let's get performance metrics on the best model
  h2o.performance(model = best_model, xval = TRUE)
  
  
  
  #### xgboost ####
  # xgboost requires a matrix input for the
  # features and the response to be a vector.
  X = as.matrix(x.train[setdiff(names(x.train), "Dx")])
  Y = x.train$Dx
  p_load(recipes)
  xgb_prep <- recipe(Dx ~ ., data = x.train) %>%
    step_integer(all_nominal()) %>%
    prep(training = x.train, retain = TRUE) %>%
    juice()
  
  X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Dx")])
  Y <- xgb_prep$Dx
  
  set.seed(123)
  ames_xgb <- xgb.cv(
    data = X,
    label = Y,
    nrounds = 6000,
    objective = "reg:squarederror",
    early_stopping_rounds = 50, 
    nfold = 10,
    params = list(
      eta = 0.1,
      max_depth = 3,
      min_child_weight = 3,
      subsample = 0.8,
      colsample_bytree = 1.0),
    verbose = 0
  )  
  
  # minimum test CV RMSE
  min(ames_xgb$evaluation_log$test_rmse_mean)
  
  
  ### next, we assess if overfitting is limiting our model's performance 
  # by performing a grid search that examines 
  # various regularization parameters (gamma, lambda, and alpha)
  # hyperparameter grid
  hyper_grid <- expand.grid(
    eta = 0.01,
    max_depth = 3, 
    min_child_weight = 3,
    subsample = 0.5, 
    colsample_bytree = 0.5,
    gamma = c(0, 1, 10, 100, 1000),
    lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
    alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
    rmse = 0,          # a place to dump RMSE results
    trees = 0          # a place to dump required number of trees
  )
  
  # grid search
  for(i in seq_len(nrow(hyper_grid))) {
    set.seed(123)
    m <- xgb.cv(
      data = X,
      label = Y,
      prediction = T,
      # callbacks = (SaveBestModel(cvboosters), ),
      nrounds = 4000,
      objective = "reg:squarederror",
      early_stopping_rounds = 50, 
      nfold = 10,
      verbose = 0,
      params = list( 
        eta = hyper_grid$eta[i], 
        max_depth = hyper_grid$max_depth[i],
        min_child_weight = hyper_grid$min_child_weight[i],
        subsample = hyper_grid$subsample[i],
        colsample_bytree = hyper_grid$colsample_bytree[i],
        gamma = hyper_grid$gamma[i], 
        lambda = hyper_grid$lambda[i], 
        alpha = hyper_grid$alpha[i]
      ) 
    )
    hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
    hyper_grid$trees[i] <- m$best_iteration
  }
  
  # results
  hyper_grid = hyper_grid %>%
    filter(rmse > 0) %>%
    arrange(rmse) 
  
  ## best model
  bst = hyper_grid[1,]

  ## fit the final model with xgb.train or xgboost
  #optimal parameter list
  params <- list(
    eta = bst[,"eta"],
    max_depth = bst[,"max_depth"],
    min_child_weight = bst[,"min_child_weight"],
    subsample = bst[,"subsample"],
    colsample_bytree = bst[,"colsample_bytree"]
  )
  
  # train final model
  xgb.fit.final <- xgboost(
    params = params,
    data = X,
    label = Y,
    nrounds = 3944,
    objective = "reg:squarederror",
    verbose = 0
  )
  
  ## top important vars
  topvar = vip::vi(xgb.fit.final)
  
  # variable importance plot
  vip::vip(xgb.fit.final)
  
}

train_matrix <- xgb.DMatrix(data = as.matrix(X), label = x.train$Dx)
test_matrix <- xgb.DMatrix(data = as.matrix(x.test[,-ncol(x.test)]), label = x.test$Dx)
pred <- predict(xgb.fit.final, test_matrix)
