# Mushroom Classification -------------------------------------------------
# The objectives of this notebooks are:
#
#- Test Tidymodels  
#- Implement a bayesian optimization approach
## Set-up ------------------------------------------------------------------
setwd("~/Google Drive/My Drive/DS_Projects/Playground_Series/ps4s8_Mushroom_Classification/")
options(scipen = 9999)

# Loading libraries
library(tidyverse)
library(tidymodels)
library(tune)
library(bonsai)
library(embed)
library(finetune)
library(future)
tidymodels_prefer()

# Lets start by preparing the target as passing characters as factors
train_ready <- read_csv("Data/train_ready_1s.csv")
test_ready <- read_csv("Data/test_ready_1s.csv")

# Modeling with Tidymodels ------------------------------------------------
# 1. Split the data
data_split <- initial_split(train_ready, prop = 0.7)
train_data <- training(data_split)
test_data <- testing(data_split)

# 2. Define two different recipes
recipe_no_encoding <- recipe(class ~ ., data = train_data) %>%
  step_rm(id) %>%
  step_YeoJohnson(all_numeric_predictors())%>%
  step_novel(all_nominal_predictors())%>%
  step_zv()

recipe_effect_encoding <- recipe(class ~ ., data = train_data) %>%
  step_rm(id) %>%
  step_YeoJohnson(all_numeric_predictors())%>%
  step_novel(all_nominal_predictors())%>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(class)) %>% 
  step_zv()

# 3. Define the LightGBM model with tunable parameters
lightgbm_model_tune <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm",
             is_unbalance = TRUE,
             early_stop = 50) %>%
  set_mode("classification")

# 4. Create a workflow set
workflow_exp <- workflow_set(
  preproc = list(no_encoding = recipe_no_encoding, effect_encoding = recipe_effect_encoding),
  models = list(lightgbm = lightgbm_model_tune),
  cross = FALSE
)

# 5. Define the resampling method
cv_folds <- vfold_cv(train_data, v = 5, strata = class)

# 6. Define metrics
metrics <- metric_set(mcc)

# 7. Define the parameter space
lgb_params <- parameters(
  trees(range = c(300, 2000)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-4, -1), trans = log10_trans()),
  min_n(range = c(30, 1000)),
  sample_size = sample_prop(range = c(0.4, 1)),
  mtry(range = c(5,21))
)

# 8. Set up Bayesian optimization
bo_ctrl <- control_bayes(
  no_improve = 20,
  time_limit = 3600,
  verbose = TRUE,
  verbose_iter = TRUE
)

# 9. Set parallel processing
plan(multisession, workers = 8)
options(future.globals.maxSize = 16.0 * 1e9)

# 10. Tune the workflows using Bayesian optimization
tuned_workflows <- workflow_exp %>%
  workflow_map(
    fn = "tune_bayes",
    resamples = cv_folds,
    iter = 100,  # Increased due to more parameters
    initial = 10,  # Increased due to more parameters
    metrics = metrics,
    control = bo_ctrl,
    param_info = lgb_params
  )
