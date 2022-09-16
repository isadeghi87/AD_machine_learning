## build classifier models 
if(T){
  
  rm(list=ls())
  library(pacman)
  p_load(dplyr,grid,caret,randomForest,pROC,ROCR,ggrepel,Rtsne,reshape2,magrittr,
         vip,gridExtra)
  
setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
 load("./data/train_test.Rdata")
  
  ## filter data for one dataset: train data
 set.seed(1234)
idx <- createDataPartition(y = dat.train$Dx, p = 0.6, list = FALSE)[,1]
 X.train <- dat.train[idx, ]
 X.test <- dat.train[-idx, ]

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
    
    
    
    ### glmnet model
    glm_model <- train(Dx ~ .,
                      x.train,
                       method = "glmnet",
                       tuneGrid = expand.grid(
                         alpha = 0:1,
                         lambda = 0:10/10),
                       trControl = ctrl1)
    
    #### knn model #
    knn_model <- train(Dx ~ .,
                       x.train,
                       method = "knn",
                       tuneLength = 20,
                       trControl = ctrl1)
    
    #### svm model #
    svm_model <- train(Dx ~ .,
                       x.train,
                       method = "svmRadial",
                       tuneLength = 10,
                       trControl = ctrl1)
    #### naive bayes #
    nb_model <- train(Dx ~ .,
                      x.train,
                      method = "naive_bayes",
                      trControl = ctrl1)
    
    ### compare models #
    model_list <- list(glmnet = glm_model,
                       rf = rf_model,
                       knn = knn_model,
                       svm = svm_model,
                       nb = nb_model)
    
    ## save data for plotting
    saveRDS(model_list,"./data/different_models.Rdata")
    
    ## choose best model 
    resamp <- caret::resamples(model_list)
    mod.sum = summary(resamp)
    mod.sum = data.frame(accuracy = mod.sum$statistics$Accuracy[,"Mean"],
                         kappa = mod.sum$statistics$Kappa[,"Mean"])
    mod.sum =  mod.sum %>% arrange(desc(accuracy))
    
    top.mod = rownames(mod.sum)[1]
    final_mod = model_list[[grep(top.mod,names(model_list))]]
    pred = predict(final_mod,x.test)
    sens = confusionMatrix(data = pred, 
                           reference = as.factor(x.test$Dx))
    
    # ## choose top important variable
    imp = varImp(final_mod)
    var = sortImp(imp,20)
    var = rownames(var)
    
    # -- ! CHECK THIS ! --
    accuracy = as.numeric(round(unname(sens$overall["Accuracy"]),2))
    kappa = as.numeric(round(unname(sens$overall["Kappa"]),2))
    p = as.numeric(signif(unname(sens$overall["AccuracyPValue"]),2))
    # sensitivity = round(unname(sens$byClass["Sensitivity"]),2)
    # specificity = round(unname(sens$byClass["Specificity"]),2)
    rm(resamp,rf_model,svm_model,glm_model,knn_model)
  
    if(!dir.exists("./results/tables")){
      dir.create("./results/tables",recursive = T)
    }
  write.csv(var,file = "./results/tables/top_important_genes.csv")
  
}

