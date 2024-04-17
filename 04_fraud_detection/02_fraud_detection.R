# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 4: FRAUD DETECTION

library(tidyverse)
library(tidymodels)
library(xgboost)

# Generate synthetic data
set.seed(0)
data_size <- 1000
transaction_types <- sample(c("type1", "type2", "type3"), size = data_size, replace = TRUE)
transaction_amounts <- rexp(n = data_size, rate = 1/200)
age_of_account_days <- rnorm(n = data_size, mean = 365, sd = 100)
fraudulent <- vector("numeric", length = data_size)

for (i in 1:data_size) {
  if (transaction_types[i] == "type1" && transaction_amounts[i] > 100 && age_of_account_days[i] < 365) {
    fraudulent[i] <- sample(c(0, 1), size = 1, prob = c(0.1, 0.9)) # 90% chance of being fraudulent
  } else {
    fraudulent[i] <- sample(c(0, 1), size = 1, prob = c(0.99, 0.01)) # 1% chance as normal
  }
}

df <- tibble(
    transaction_amount = transaction_amounts,
    transaction_type = transaction_types,
    age_of_account_days = age_of_account_days,
    fraudulent = as.factor(fraudulent)
) %>% rowid_to_column()

df_untransformed <- df

# Data preparation
df <- df %>%
  mutate(transaction_type = as.numeric(as.factor(transaction_type))) # Encode categorical data as numeric

# Splitting the data
set.seed(0)
split <- initial_split(df, prop = 0.8)
train_data <- training(split)
test_data <- testing(split)

# XGBoost Model Setup
xgb_spec <- boost_tree(trees = 50, tree_depth = 6, min_n = 10) %>%
  set_engine("xgboost", eval_metric = "logloss") %>%
  set_mode("classification")

# Fit the Model
xgb_fit <- xgb_spec %>%
  fit(fraudulent ~ . - rowid, data = train_data)

# Make Predictions
test_results <- test_data %>%
  select(-fraudulent) %>%
  predict(xgb_fit, new_data = ., type = "prob") %>%
  mutate(predict_class = as.numeric(.pred_1 > 0.5) %>% as.factor()) %>%
  bind_cols(
    df_untransformed %>% filter(rowid %in% test_data$rowid)
  ) 

# Evaluate the Model
test_results %>%
  select(fraudulent, predict_class) %>%
  yardstick::metrics(truth = fraudulent, estimate = predict_class)

# Confusion Matrix
conf_mat(test_results, truth = fraudulent, estimate = predict_class)

# Visualization
test_results %>%
  ggplot(aes(x = transaction_amount, fill = as.factor(fraudulent))) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  scale_fill_manual(values = c("gray", "red"), labels = c("Legitimate", "Fraudulent")) +
  labs(title = "Transaction Amount Distribution",
       x = "Transaction Amount",
       y = "Frequency",
       fill = "Transaction Type") +
  theme_minimal()
