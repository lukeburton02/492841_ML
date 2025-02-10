# ML assessment. Candidate number: 492841

# Set your working directory
setwd("C:/Users/lukeb/Downloads/LSHTM/TERM 2/Machine Learning/ML assessment/492841_ML")

# Import relevant libraries
install.packages("mboost")
library(tidyverse)
library(dplyr)
library(ggplot2)
library(glmnet)
library(mboost)
library(xgboost)
library(caret)
library(pROC)

# Read in the dataset. Empty rows are ommitted by default
dat <- read.csv("assignment2025.csv")
summary(dat)
# View the first 5 rows
head(dat, 5)

# Check for missing values
sum(is.na(dat)) # No missing values

# Create a new dataframe with categorical variables converted
categorical_cols <- names(dat)[sapply(dat, function(x) is.character(x) | is.factor(x))]
categorical_cols <- setdiff(categorical_cols, 'death')
dat[categorical_cols] <- lapply(dat[categorical_cols], factor)

# ------------------------------------
# Section 1: Exploratory data analysis
# ------------------------------------

summary(dat)

table(y)

tstats <- apply(data[, -which(names(data) == "death")], 2, function(x) {
  t.test(x ~ data$death)$statistic
})
barplot(tstats, las = 2, main = "T-Statistics for Each Predictor", 
        col = "black", border = "red", ylim = c(-30, 30))
abline(h = c(-2, 2), col = "red", lwd = 2)





# Explore death rate by subtype
death_rate <- dat %>%
  group_by(subtype) %>%
  summarise(drate = mean(death) * 100)

ggplot(death_rate, aes(x = subtype, y = drate)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  labs(x = "Stroke subtype", y = "Percentage Who Died (%)", title = "Death Rate by Stroke Subtype") +
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

# L2 boosting - gradient boosting with logistic loss
# Convert the outcome to a factor, bearing in mind coefficients are halved under glmboost
dat$death <- as.factor(dat$death) 
l2boost <- glmboost(death ~ ., data=dat, family = Binomial())

# Explore coefficient evolution
boostcoefs <- coef(l2boost, aggregate = "cumsum") |> 
  do.call(what = rbind) |> t() 
colnames(boostcoefs) <- names(coef(l2boost, aggregate = "cumsum"))

# Increase iterations to identify optimal iterations
l2boost2 <- glmboost(death ~ ., data=dat, control = boost_control(mstop = 1000))
set.seed(2) # For reproducibility
folds <- cv(model.weights(l2boost2), type = "kfold") # Define CV folds
cvres2 <- cvrisk(l2boost2, folds) # Apply CV
plot(cvres2)

# Change the learning rate to 0.5
l2boost3 <- glmboost(death ~ ., data = dat, control = boost_control(mstop = 1000, nu = .5))
cvres3 <- cvrisk(l2boost3, folds) # Apply CV
plot(cvres3)





# XGBoost - save this for later

set.seed(42)  # For reproducibility
# Ensure death is numeric (0,1)
dat$death <- as.numeric(as.character(dat$death))  # Convert factor to numeric

# Split into training and validation sets before encoding
trainIndex <- createDataPartition(dat$death, p = 0.8, list = FALSE)
trainData <- dat[trainIndex, ]
validData <- dat[-trainIndex, ]

# One-hot encode categorical variables using model.matrix()
train_matrix <- model.matrix(death ~ . -1, data = trainData)  # Remove intercept
valid_matrix <- model.matrix(death ~ . -1, data = validData)

# Create XGBoost data matrices
xgb_train <- xgb.DMatrix(data = train_matrix, label = trainData$death)
xgb_valid <- xgb.DMatrix(data = valid_matrix, label = validData$death)

# Define XGBoost parameters
params <- list(
  objective = "binary:logistic",  # Probability estimation for binary classification
  eval_metric = "logloss",        # Log-loss is best for probability prediction
  eta = 0.1,                      # Learning rate
  max_depth = 4,                  # Tree depth
  subsample = 0.8,                # Prevent overfitting
  colsample_bytree = 0.8,          # Feature sampling for diversity
  scale_pos_weight = sum(trainData$death == 0) / sum(trainData$death == 1) # Weight due to class imbalance
)

# Train XGBoost model with early stopping
xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 500,                   # Number of boosting rounds
  watchlist = list(train = xgb_train, valid = xgb_valid),
  early_stopping_rounds = 10,       # Stops if no improvement
  verbose = 1
)

# Get probability predictions
pred_probs <- predict(xgb_model, newdata = xgb_valid)

# Convert probabilities to binary predictions (threshold = 0.5)
pred_labels <- ifelse(pred_probs > 0.5, 1, 0)

# Confusion Matrix & Performance Metrics
conf_matrix <- confusionMatrix(as.factor(pred_labels), as.factor(validData$death))
print(conf_matrix)

# ROC Curve & AUC Score
roc_curve <- roc(validData$death, pred_probs)
print(auc(roc_curve))  # AUC score
plot(roc_curve, col = "blue", main = "ROC Curve for XGBoost Model")



# ------------------------------------
# Section 4: Comparison of models
# ------------------------------------