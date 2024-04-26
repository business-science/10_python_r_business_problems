library(tidyverse)
library(tidymodels)
library(xgboost)

# Generate synthetic data
set.seed(0)
data_size <- 1000
transaction_types <- sample(c("type1", "type2", "type3"), size = data_size, replace = TRUE)
transaction_amounts <- rexp(n = data_size, rate = 1 / 200)
age_of_account_days <- rnorm(n = data_size, mean = 365, sd = 100)
interaction_term <- transaction_amounts * age_of_account_days  # New feature: Interaction term
fraudulent <- vector("numeric", length = data_size)

for (i in 1:data_size) {
  if (transaction_types[i] == "type1" && transaction_amounts[i] > 100 && age_of_account_days[i] < 365) {
    fraudulent[i] <- sample(c(0, 1), size = 1, prob = c(0.1, 0.9))
  } else {
    fraudulent[i] <- sample(c(0, 1), size = 1, prob = c(0.99, 0.01))
  }
}

df <- tibble(
  transaction_amount = transaction_amounts,
  transaction_type = transaction_types,
  age_of_account_days = age_of_account_days,
  interaction_term = interaction_term,
  fraudulent = as.factor(fraudulent)
)

# Data preparation
df <- df %>%
  mutate(transaction_type = as.numeric(as.factor(transaction_type)))

# Split the data
set.seed(0)
split <- initial_split(df, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)

# Model specification with hyperparameter tuning
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost", eval_metric = "logloss") %>%
  set_mode("classification")

# Workflow setup
xgb_wf <- workflow() %>%
  add_model(xgb_spec) %>%
  add_formula(fraudulent ~ .)

# Tuning parameters
xgb_grid <- grid_random(
  trees = c(50, 100),
  tree_depth = c(6, 10),
  min_n = c(10, 20),
  loss_reduction = c(0, 0.01),
  sample_size = c(0.8, 1.0),
  mtry = c(2, 3),
  learn_rate = c(0.01, 0.1),
  size = 30
)

# Tune the model
set.seed(0)
xgb_results <- tune_grid(
  xgb_wf,
  resamples = vfold_cv(train_data, v = 5),
  grid = xgb_grid
)

# Best model selection
best_params <- select_best(xgb_results, "logloss")

# Final model training
final_xgb_model <- finalize_workflow(
  xgb_wf,
  best_params
) %>%
  last_fit(split)

# Results
final_xgb_model
