# ML assessment. Candidate number: 492841

# Set your working directory
setwd("C:/Users/lukeb/Downloads/LSHTM/TERM 2/Machine Learning/ML assessment/492841_ML")

# Import relevant libraries
library(tidyverse)
library(dplyr)
library(ggplot2)
library(glmnet)

# Read in the dataset. Empty rows are ommitted by default
data <- read.csv("assignment2025.csv")

# View the first 5 rows
head(data, 5)

# Check for missing values
sum(is.na(data)) # No missing values

# Create a new dataframe with 

# ------------------------------------
# Section 1: Exploratory data analysis
# ------------------------------------

summary(data)

table(y)

tstats <- apply(data[, -which(names(data) == "death")], 2, function(x) {
  t.test(x ~ data$death)$statistic
})
barplot(tstats, las = 2, main = "T-Statistics for Each Predictor", 
        col = "black", border = "red", ylim = c(-30, 30))
abline(h = c(-2, 2), col = "red", lwd = 2)


# Explore death rate by subtype
death_rate <- data %>%
  group_by(subtype) %>%
  summarise(drate = mean(death) * 100)

ggplot(death_rate, aes(x = subtype, y = drate)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  labs(x = "Stroke subtype", y = "Percentage who died (%)", title = "Death Rate by Stroke Subtype") +
  theme_minimal()

# ------------------------------------
# Split into training + validation
# ------------------------------------





# ------------------------------------
# Section 2: Regularised regression
# ------------------------------------








# ------------------------------------
# Section 3: Tree-based methods
# ------------------------------------










# ------------------------------------
# Section 4: Comparison of models
# ------------------------------------