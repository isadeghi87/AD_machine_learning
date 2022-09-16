## building deep learning model
if(T){
  
  rm(list=ls())
  pacman::p_load(dplyr,
         grid,
         foreach,
         doParallel,
         gridExtra,
         ggplot2,
         magrittr,
         keras,# for fitting DNNs
         tfruns, #for additional grid search & model training functions
         tfestimators,# provides grid search & model training interface
         vip)        
  
  setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
  load("./data/train_test.Rdata")
  
  ##eedforward DNNs require all feature inputs to be numeric. 
  #if data contains categorical features they will need to be 
  #numerically encoded (e.g., one-hot encoded, integer label encoded, etc.).
  
  ## a one-hot encoded matrix, which can be accomplished with 
  ## the keras function to_categorical()
  
  # One-hot encode response
  colnames(x) <- paste0("V", 1:ncol(mnist_x))  
  Y <- to_categorical(as.numeric(x.train$Dx),num_classes = 3)