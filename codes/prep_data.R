## here we prepare the data for training 
if(T){
library(pacman)
p_load(dplyr,reshape2,magrittr,WGCNA,limma,rsample)

setwd("C:/Users/crgcomu/Desktop/Iman/Brain_meta/projects/AD_machine_learning/")
load("C:/Users/crgcomu/Desktop/Iman/Brain_meta/projects/condition_overlap/disease_specific/AD_metaAnalysis/AD_normalized.Rdata")

  ## filter data for one dataset: train data
  id = which(datMeta$Study=="Mayo")
  meta = datMeta[id,]
  meta$Brain_Lobe = factor(meta$Brain_Lobe)
  exp = datExp[,id]
  
  ## remove near zero variance genes
  nzv = caret::nearZeroVar(t(exp), saveMetrics = TRUE) %>% 
    tibble::rownames_to_column() %>% 
    filter(nzv)
  id = nzv$rowname
  exp = exp[setdiff(rownames(exp),id),]
  
  ## batch correction
  dt = sva::ComBat_seq(exp,batch = meta$Brain_Lobe,group = meta$Dx)
  mod = model.matrix(~0+Dx,data = meta)
  v = limma::voom(dt,mod)
  datExp = v$E
  
  ## Remove outliers based on network connectivity z-scores
  normadj <- (0.5+0.5*bicor(datExp))^2 ## Calculate connectivity
  netsummary <- fundamentalNetworkConcepts(normadj)
  ku <- netsummary$Connectivity
  z.ku <- (ku-mean(ku))/sqrt(var(ku))
  outliers = (z.ku < -2)
  table(outliers)
  
  exp = v$E[,!outliers]
  meta = meta[!outliers,]
  
  ## save data
  normalized = list(exp,meta)
  names(normalized)  = c("datExp","datMeta")
  
  ## merge data for training
  Dx = meta[,"Dx",drop=F]
  df = cbind(t(exp),Dx)
  
  # Create training (70%) and test (30%) sets for the data.
  # Use set.seed for reproducibility
  set.seed(123)
  churn_split <- rsample::initial_split(df, prop = .7, strata = "Dx")
  x.train <- rsample::training(churn_split)
  x.test  <- rsample::testing(churn_split)

  save(file = "./data/train_test.Rdata",df,normalized,x.train,x.test)
}

