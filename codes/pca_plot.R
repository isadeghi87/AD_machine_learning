## here we perorm PCA and MDS for train and test data 
## using all genes and top important genes
setwd("/users/rg/isadeghi/projects/AD_machine_learning/")
load("./data/train_test.Rdata")
gene = read.csv("./results/tables/top_important_genes.csv,row.names=1")
colnames(gene)="var"


exp  = normalized[["datExp"]]
meta  = normalized[["datMeta"]]

pc = prcomp(t(exp))
p1 = factoextra::fviz_pca_ind(pc,col.ind = meta$Dx,label="none")+
  scale_color_brewer(palette="Set1")+ 
  labs(title = paste(n[i],": all genes"))+
  theme_bw()+
  theme(panel.grid = element_blank())  

## redo PCA using top important genes
subexp = exp[gene,]
pc = prcomp(t(subexp))
p2 = factoextra::fviz_pca_ind(pc,col.ind = meta$Dx,label="none")+
  scale_color_brewer(palette="Set1")+ 
  labs(title = paste(n[i],": top important genes"))+
  theme_bw()+
  theme(panel.grid = element_blank())  

gg = ggpubr::ggarrange(p1,p2,nrow= 1, common.legend = T)


if(!dir.exists("./results/figures/pca")){
  dir.create("./results/figures/pca",recursive = T)
}
ggsave(filename = "./results/figures/pca/mds_plots.pdf",plot = gg,
       width = 10, height = 5)