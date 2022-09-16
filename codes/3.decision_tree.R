## building decision trees models
if(T){
  
  rm(list=ls())
  library(pacman)
  p_load(dplyr,
         grid,
         gridExtra,ggplot2,
         caret,# meta engine for decision tree application
         randomForest,
         magrittr,
         ranger,# a c++ implementation of random forest
         h2o,# a java-based implementation of random forest
         vip, # for important variables
         rpart,       # direct engine for decision tree application
         # Model interpretability packages
         rpart.plot,  # for plotting decision trees
         vip,         # for feature importance
         pdp)         # for feature effects)
  
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
  
  
  dt1 <- rpart(
    formula = Dx ~ .,
    data    = x.train,
    method  = "anova"
  )
  
  ## visulaize 
 pdf("./results/figures/decision_tree/rpart.plot.pdf")
  rpart.plot(dt1,cex = 0.5)
  plotcp(dt1)
  dev.off()
  
  dt2 <- rpart(
    formula = Dx ~ .,
    data    = x.train,
    method  = "anova", 
    control = list(cp = 0, xval = 10)
  )
  
  plotcp(dt2)
 dt1$cptable
  
 # caret cross validation results
 dt3 <- train(
   Dx ~ .,
   data = x.train,
   method = "rpart",
   trControl = trainControl(method = "cv", number = 10),
   tuneLength = 20
 )
 
 p = ggplot(dt3)+
   theme_bw()+r
   labs(title= "decision tree cross validation")
 
 ggsave("./results/figures/decision_tree.pdf",plot = p)
 
 ## feature importance
 vip(dt3, num_features = 40, bar = F)
  vi <- varImp(dt3)
  write.csv(vi,file="./results/tables/decisionTree_varImp.csv")

  ## save data for plotting
  saveRDS(model_list,"./data/knn_model.Rdata")
  
 
}

