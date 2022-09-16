## here we plot ROC curve and parameters from different models
setwd("/users/rg/projects/AD_machine_learning")

library(pacman)
p_load(dplyr,grid,caret,randomForest,pROC,ROCR,reshape2,magrittr,ggpubr,ggthemes)

## load models
load("./data/different_models.Rdata")

n = length(model_list)
modname = names(model_list)

for(i in modname){
  mod = model_list[[i]]
  
  ## plot the model
  p = ggplot(mod)+
    theme_bw()+
    labs(title = names(model_list[i]))
  modplots[[i]] = p
  rm(p)
  
  ## prediction and probabilites
  m_prob = predict(mod, x.train, type = "prob")[,2]
  pred = prediction(m_prob, x.train$Dx)
  roc = performance(pred, measure = "tpr", x.measure = "fpr")
  perd_AUC=performance(pred,"auc")
  AUC=perd_AUC@y.values[[1]]
  
  prob[[i]]$roc = roc
  prob[[i]]$auc = AUC
}

g =  ggpubr::ggarrange(plotlist = modplots,nrow= 2,ncol=3)
mod_pdf = "./results/figures/models_plot.pdf"
ggsave(filename = mod_pdf,plot=g,height = 5,width = 10,)

# Plot ROC curves 
roc_pdf = "./results/figures/models_ROC_plot.pdf"
pdf(roc_pdf,width = 6,height = 4)

plot(prob$nb$roc, col = "black", lty = 1)
for(i in 2:4){
  plot(prob[[i]]$roc, add = TRUE, col =i,lty = 1,cex=0.9)
}
## add legend
legend("center", legend = names(model_list),
       col=1:n ,lty = 1, cex = 0.7)
dev.off()



