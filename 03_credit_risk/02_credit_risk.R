# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 3: CREDIT RISK

# Install and load necessary packages
library(tidyverse)
library(tidymodels)

# Set seed for reproducibility
set.seed(0)

# 2. Generate Data
data_size <- 1000
credit_scores <- rnorm(data_size, mean=600, sd=100)
annual_incomes <- rnorm(data_size, mean=50000, sd=15000)
credit_risks <- ifelse(credit_scores < 580 | annual_incomes < 30000, 1, 0) # Simplified risk criteria

# Create a tibble
data <- tibble(
  credit_score = credit_scores,
  annual_income = annual_incomes,
  credit_risk = factor(credit_risks, levels = c(0, 1))
)

# 3. Preprocess Data
split <- initial_split(data, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)

# Recipe for preprocessing
recipe <- recipe(credit_risk ~ ., data = train_data) %>%
  step_scale(all_predictors()) %>%
  step_center(all_predictors())

# 4. Train a Model
logit_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(logit_model) %>%
  fit(data = train_data)

# 5. Evaluate the Model
test_results <- workflow %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data) %>%
  metrics(truth = credit_risk, estimate = .pred_class)

accuracy <- test_results %>%
  filter(.metric == "accuracy")

cat("Accuracy:", accuracy$.estimate, "\n")
conf_mat <- workflow %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data) %>%
  conf_mat(truth = credit_risk, estimate = .pred_class)

print(conf_mat)
