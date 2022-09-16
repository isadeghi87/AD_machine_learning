## building random forest models
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
  ctrl1 <- trainControl(method = "cv",
                        number = 5,
                        # summaryFunction = twoClassSummary,
                        verboseIter = TRUE,
                        allowParallel = T,
                        savePredictions = TRUE,
                        classProbs  = TRUE)
  
  #### random forest model#
  rf_model <- train(Dx ~ ., data = x.train, 
                    method = "rf",
                    tuneGrid=myGrid,
                    trControl = ctrl1)
  
  
  # number of features
  n_features <- length(setdiff(names(x.train), "Dx"))
  
  # train a default random forest model
  ames_rf1 <- ranger(
    Dx ~ ., 
    data = x.train,
    mtry = floor(n_features / 3),
    respect.unordered.factors = "order",
    seed = 123
  )
  
  # get OOB RMSE
  (default_rmse <- sqrt(ames_rf1$prediction.error))

  # Tuning strategies ####
  # create hyperparameter grid
  hyper_grid <- expand.grid(
    mtry = floor(n_features * c(.05, .15, .25, .333, .4)),
    min.node.size = c(1, 3, 5, 10), 
    replace = c(TRUE, FALSE),                               
    sample.fraction = c(.5, .63, .8),                       
    rmse = NA                                               
  )
  
  # execute full cartesian grid search
  for(i in seq_len(nrow(hyper_grid))) {
    # fit model for ith hyperparameter combination
    fit <- ranger(
      formula         = Dx ~ ., 
      data            = x.train, 
      num.trees       = n_features * 10,
      mtry            = hyper_grid$mtry[i],
      min.node.size   = hyper_grid$min.node.size[i],
      replace         = hyper_grid$replace[i],
      sample.fraction = hyper_grid$sample.fraction[i],
      verbose         = FALSE,
      seed            = 123,
      respect.unordered.factors = 'order',
    )
    # export OOB error 
    hyper_grid$rmse[i] <- sqrt(fit$prediction.error)
  }
  
  # assess top 10 models
  hyper_grid = hyper_grid %>%
    arrange(rmse) %>%
    mutate(perc_gain = (default_rmse - rmse) / default_rmse * 100) 
  head(hyper_grid,10)
  
  
  ###  fit a random forest model with h2o ####
  h2o.no_progress()
  h2o.init(max_mem_size = "5g")

  # convert training data to h2o object
  train_h2o <- as.h2o(x.train)
  
  # set the response column to Sale_Price
  response <- "Dx"
  
  # set the predictor names
  predictors <- setdiff(colnames(x.train), response)
  
  #The following fits a default random forest model with h2o to illustrate that our baseline results ( 
  # are very similar to the baseline ranger model we fit earlier.
  h2o_rf1 <- h2o.randomForest(
    x = predictors, 
    y = response,
    training_frame = train_h2o, 
    ntrees = n_features * 10,
    seed = 123)
  
  h2o_rf1
  
  # To execute a grid search in h2o we need our hyperparameter grid to be a list.
  # For example, the following code searches a larger grid space than before with
  # a total of 240 hyperparameter combinations. We then create a random grid 
  # search strategy that will stop if none of the last 10 models have managed
  # to have a 0.1% improvement in MSE compared to the best model before that.
  # If we continue to find improvements then we cut the grid search off after 300 
  # seconds (5 minutes).
  
  # hyperparameter grid
  hyper_grid <- list(
    mtries = floor(n_features * c(.05, .15, .25, .333, .4)),
    min_rows = c(1, 3, 5, 10),
    max_depth = c(10, 20, 30),
    sample_rate = c(.55, .632, .70, .80)
  )
  
  # random grid search strategy
  search_criteria <- list(
    strategy = "RandomDiscrete",
    stopping_metric = "mse",
    stopping_tolerance = 0.001,   # stop if improvement is < 0.1%
    stopping_rounds = 20,         # over the last 10 models
    max_runtime_secs = 60*5      # or stop search after 5 min.
  )
  
  # perform grid search 
  random_grid <- h2o.grid(
    algorithm = "randomForest",
    grid_id = "rf_random_grid",
    x = predictors, 
    y = response, 
    training_frame = train_h2o,
    hyper_params = hyper_grid,
    ntrees = n_features * 10,
    seed = 123,
    stopping_metric = "RMSE",   
    stopping_rounds = 10,           # stop if last 10 trees added 
    stopping_tolerance = 0.005,     # don't improve RMSE by 0.5%
    search_criteria = search_criteria
  )
  
  #collect the results and sort by our model performance metric 
  # of choice
  random_grid_perf <- h2o.getGrid(
    grid_id = "rf_random_grid", 
    sort_by = "mse", 
    decreasing = FALSE
  )
  
  bst = random_grid_perf@summary_table[1,]
  best_model_id <- bst$model_ids
  best_model <- h2o.getModel(best_model_id)
  
  # Now let's get performance metrics on the best model
  perf = h2o.performance(model = best_model, xval = TRUE)
  perf
  
  
  # re-run model with impurity-based variable importance
  rf_impurity <- ranger(
    formula = Dx ~ ., 
    data = x.train, 
    num.trees = bst$max_depth,
    mtry = bst$mtries,
    min.node.size = 1,
    sample.fraction = bst$sample_rate,
    replace = FALSE,
    importance = "impurity",
    respect.unordered.factors = "order",
    verbose = FALSE,
    seed  = 123
  )
  
  # re-run model with permutation-based variable importance
  rf_permutation <- ranger(
    formula = Dx ~ ., 
    data = x.train, 
    num.trees = bst$max_depth,
    mtry = bst$mtries,
    min.node.size = 1,
    sample.fraction = bst$sample_rate,
    replace = FALSE,
    importance = "permutation",
    respect.unordered.factors = "order",
    verbose = FALSE,
    seed  = 123
  )
  
  ## look at the top important vars
  p1 <- vip::vip(rf_impurity, num_features = 25, bar = FALSE,
                 aesthetics = list(fill = "blue",color = "black"))+
    theme_bw()
  p2 <- vip::vip(rf_permutation, num_features = 25, bar = FALSE,
  aesthetics = list(fill = "green",color = "black"))+
  theme_bw()
  
  gridExtra::grid.arrange(p1, p2, nrow = 1)
  
  ## save data for plotting
  saveRDS(model_list,"./data/rf_model.Rdata")
  
 
  if(!dir.exists("./results/tables")){
    dir.create("./results/tables",recursive = T)
  }
  write.csv(var,file = "./results/tables/top_important_genes.csv")
  
}

