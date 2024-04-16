# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 1: CUSTOMER SEGMENTATION

library(tidyverse)

# Creating sample data
data <- tibble(
  Age = c(25, 47, 35, 45, 22, 34, 52, 23, 40, 60),
  Annual_Income_k = c(25, 60, 29, 55, 20, 40, 50, 15, 60, 30),
  Spending_Score = c(30, 55, 35, 50, 45, 50, 30, 25, 70, 40)
)

# Performing K-means clustering
set.seed(42) # for reproducibility
clusters <- kmeans(data, centers = 3, nstart = 25)

# Adding cluster results to the data frame
data$Cluster <- as.factor(clusters$cluster)

# Plotting the clusters
ggplot(data, aes(x = Age, y = Annual_Income_k, color = Cluster)) +
  geom_point(aes(size = Spending_Score), alpha = 0.6) +
  scale_color_manual(values = c("red", "green", "blue")) +
  labs(title = "Customer Segmentation with K-means Clustering",
       x = "Age",
       y = "Annual Income (k$)",
       color = "Cluster") +
  theme_minimal()
