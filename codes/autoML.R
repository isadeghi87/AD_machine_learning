## here we perform automated ML
pacman::p_load(dplyr,magrittr,rsample,recipes,h2o)

## load data
load("./data/train_test.Rdata")
nc = ncol(x.train)
x.train  = x.train[,(nc-50):nc]

X <- model.matrix(Dx ~ ., x.train)[, -1]
Y <- x.train$Dx

# Make sure we have consistent categorical levels
blueprint <- recipe(Dx ~ ., data = x.train) %>%
  step_other(all_nominal(), threshold = 0.005)

# Create training & test sets for h2o
h2o.init(min_mem_size = "5g")
train_h2o <- prep(blueprint, training = x.train, retain = TRUE) %>%
  juice() %>%
  as.h2o()
test_h2o <- prep(blueprint, training = x.train) %>%
  bake(new_data = x.test) %>%
  as.h2o()

# Get response and feature names
Y <- "Dx"
X <- setdiff(names(x.train), Y)

# Use AutoML to find a list of candidate models
auto_ml <- h2o.automl(
  x = X, y = Y, training_frame = train_h2o, nfolds = 5, 
  max_runtime_secs = 60 * 120, max_models = 50,
  keep_cross_validation_predictions = TRUE, sort_metric = "RMSE", seed = 123,
  stopping_rounds = 50, stopping_metric = "RMSE", stopping_tolerance = 0
)

# Assess the leader board; the following truncates the results to show the top 
# and bottom 15 models. You can get the top model with auto_ml@leader
auto_ml@leaderboard %>% 
  as.data.frame() %>%
  dplyr::select(model_id, rmse) %>%
  dplyr::slice(1:25)
