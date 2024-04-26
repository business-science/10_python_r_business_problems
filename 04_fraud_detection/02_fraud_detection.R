library(tidyverse)
library(tidymodels)
library(xgboost)
library(caret)  # For improved cross-validation

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

# Model setup with hyperparameter tuning and cross-validation
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = c(6, 8, 10),
  eta = c(0.01, 0.05, 0.1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1
)

train_control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE
)

xgb_train <- train(
  fraudulent ~ .,
  data = df,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid
)

# Evaluate the model
print(xgb_train)
