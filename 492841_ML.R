# ML assessment. Candidate number: 492841

# Set your working directory
setwd("C:/Users/lukeb/Downloads/LSHTM/TERM 2/Machine Learning/ML assessment/492841_ML")

# Import relevant libraries
install.packages("mboost")
install.packages("GGally") # remove if unused
library(tidyverse)
library(dplyr)
library(ggplot2)
library(glmnet)
library(mboost)
library(xgboost)
library(caret) # used for training, predicting, and pre-processing
library(pROC)
library(ggcorrplot)
library(GGally)
library(gridExtra)
library(forcats) # for ordering bar plots

# Read in the dataset. Empty rows are ommitted by default
dat <- read.csv("assignment2025.csv")
summary(dat)
# View the first 5 rows
head(dat, 5)

# Check for missing values
sum(is.na(dat)) # No missing values

# Convert categorical variables to factors
categorical_cols <- names(dat)[sapply(dat, function(x) is.character(x) | is.factor(x))]
categorical_cols <- setdiff(categorical_cols, 'death')
dat[categorical_cols] <- lapply(dat[categorical_cols], factor)

# Convert the outcome to a factor for proper handling
dat$death <- factor(dat$death, levels = c(0, 1), labels = c("Survived", "Died"))

# ------------------------------------
# Section 1: Exploratory data analysis
# ------------------------------------

summary(dat)







# Explore death rate by subtype
death_rate <- dat %>%
  group_by(subtype) %>%
  summarise(drate = mean(as.numeric(death)-1) * 100)

ggplot(death_rate, aes(x = fct_reorder(subtype, drate), y = drate)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  labs(x = "Stroke Subtype", y = "Percentage Who Died (%)", title = "Death Rate by Stroke Subtype") +
  theme_minimal() +
  theme_bw()


# 2. Outcome variable distribution (Bar plot)
ggplot(dat, aes(x = death)) +
  geom_bar(fill = "blue") +
  labs(title = "Death Outcome Distribution", x = "Death (0 = Survived, 1 = Died)", y = "Count")

# 3. Boxplots for numeric variables by death outcome
p1 <- ggplot(dat, aes(x = death, y = delay, fill = death)) +
  geom_boxplot() + labs(title = "Delay Time by Death Outcome")

p2 <- ggplot(dat, aes(x = death, y = age, fill = death)) +
  geom_boxplot() + labs(title = "Age by Death Outcome")

p3 <- ggplot(dat, aes(x = death, y = sbp, fill = death)) +
  geom_boxplot() + labs(title = "Systolic BP by Death Outcome")

grid.arrange(p1, p2, p3, ncol = 3)

# 4. Bar plots for categorical variables by death outcome
cat_summary <- dat %>%
  select(all_of(categorical_cols), death) %>%
  select(-starts_with("symptom")) %>%
  group_by(death) %>%
  summarise(across(everything(), ~paste0(sum(. == levels(.)[2]), " (", round(mean(. == levels(.)[2]) * 100, 1), "%)"))) %>%
  pivot_longer(-death, names_to = "Variable", values_to = "Count (%)")

print(cat_summary)  # Display summary table

cat_summary <- dat %>%
  select(all_of(categorical_cols), death) %>%
  select(-starts_with("symptom")) %>%
  group_by(death) %>%
  summarise(across(everything(), ~sum(. == levels(.)[2]))) %>%
  pivot_longer(-death, names_to = "Variable", values_to = "Count") %>%
  pivot_wider(names_from = death, values_from = Count)
colnames(cat_summary) <- c("Variable", "Survived", "Died")
print(cat_summary)

# 5. Correlation analysis (only binary categorical + numeric vars with death)
binary_cats <- categorical_cols[sapply(dat[categorical_cols], function(x) length(unique(x)) == 2)]
selected_vars <- c(numeric_cols, binary_cats)

# Convert binary categorical variables to numeric for correlation
dat_corr <- dat %>% select(all_of(selected_vars)) %>% mutate(across(all_of(binary_cats), as.numeric))
dat_corr$death <- as.numeric(dat$death)  # Convert death to numeric

# Compute correlation matrix
cor_matrix <- cor(dat_corr, use = "complete.obs")
cor_values <- data.frame(Variable = rownames(cor_matrix)[-ncol(cor_matrix)], 
                         Correlation = round(cor_matrix[-ncol(cor_matrix), "death"], 3))
cor_values <- cor_values[order(abs(cor_values$Correlation), decreasing = TRUE), ]  # Sort by absolute correlation

print(cor_values)  # Display correlation table


# 6. Stacked bar plot for symptom variables by death outcome
symptom_vars <- grep("^symptom", names(dat), value = TRUE)
symptom_death <- dat %>%
  pivot_longer(cols = all_of(symptom_vars), names_to = "Symptom", values_to = "Present") %>%
  filter(Present == "Y") %>%
  group_by(Symptom) %>%
  summarise(DeathRate = mean(death == "Died") * 100)

ggplot(symptom_death, aes(x = reorder(Symptom, DeathRate), y = DeathRate, fill = DeathRate)) +
  geom_col() +
  coord_flip() +
  labs(title = "Death Rate by Symptom", y = "Death Rate (%)", x = "Symptom") +
  scale_fill_gradient(low = "blue", high = "red")

# ------------------------------------
# Split into training + validation
# ------------------------------------

# We use createDataPartition to preserve the death distribution, reducing bias

psamp <- .2 # Use an 80-20 train-test split
set.seed(22) # For reproducibility
testind <- createDataPartition(y = dat$death, p = psamp)[[1]]

# Split data
dattr <- dat[-testind,]
datte <- dat[testind,]


# Note that we will not oversample on the validation dataset
# This is to preserve the underlying low death rate of ~5%

# Convert to model matrix form to use in models
Xtr <- model.matrix(death ~ . - 1, dattr)
Ytr <- dattr$death
Xte <- model.matrix(death ~ . - 1, datte)
Yte <- datte$death

# ------------------------------------
# Section 2: Regularised regression
# ------------------------------------

# Use glmnet to create a ridge regression model
ridgemod <- cv.glmnet(Xtr, Ytr,
                      family = "binomial",
                      alpha = 0) # Ridge regression

# Plot the binomial deviance for different values of lambda
plot(ridgemod, main = "Ridge Model Tuning")



# Train a lasso model with alpha = 1 (default)
lassomod <- cv.glmnet(Xtr, Ytr,
                      family = "binomial")
plot(lassomod, main = "Lasso tuning")



# Train an elastic net model with 0 < alpha < 1
enetmod <- train(death ~ .,
                 data = dattr,
                 method = "glmnet", 
                 family = "binomial",
                 tuneGrid = expand.grid(alpha = 0:10 / 10, 
                                        lambda = 10^seq(-5, 2, length.out = 100)),
                 trControl = trainControl("cv",
                                          number = 10,
                                          summaryFunction = twoClassSummary,
                                          classProbs = TRUE),
                 metric = "ROC")
enetmod$bestTune

# Plot the ROC for each model
# We see that alpha = 0, corresponding to ridge, performs best
plot(enetmod, xTrans = log10)


# Try minimising log-loss instead of maximising AUC
enetmod2 <- train(death ~ ., dattr, method = "glmnet", 
                  family = "binomial",
                  tuneGrid = expand.grid(alpha = 0:10 / 10, 
                                         lambda = 10^seq(-5, 2, length.out = 100)),
                  trControl = trainControl("cv", classProbs = TRUE, summaryFunction = mnLogLoss), 
                  metric = "logLoss")
enetmod2$bestTune
plot(enetmod2, xTrans = log10)

# ------------------------------------
# Section 3: Tree-based methods
# ------------------------------------

# L2 boosting - gradient boosting with logistic loss
# Convert the outcome to a factor, bearing in mind coefficients are halved under glmboost
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





# XGBoost

# Extract data as matrices
#X <- model.matrix(death ~ . - 1, dattr)
#Y <- dattr$death

# Try upsampling
up_data <- upSample(x = dattr %>% select(-death), y = dattr$death) %>%
  rename(death = Class)

X <- model.matrix(death ~ . - 1, up_data)
Y <- factor(up_data$death, levels = c("Survived", "Died"), labels = c("Survived", "Died"))

# Convert to DMatrix - UPDATE NAME
xgdata <- xgb.DMatrix(data = X, label = as.numeric(Y) - 1)

# Tune the number of iterations
set.seed(55)
params <- list(objective = "binary:logistic", eval_metric = "auc",
               scale_pos_weight = 2, # pay more attention to minority class to improve specificity
               max_depth = 6,
               min_child_weight = 8,
               subsample = 0.8,
               colsample_bytree = 0.8,
               eta = 0.05, # control the learning rate
               gamma = 2, # reduction required before a split is made
               lambda = 1.5,
               alpha = 0.5) # controls regularisation
xgcv <- xgb.cv(data = xgdata, nrounds = 100, nfold = 6, verbose = F, params = params,
               early_stopping_rounds = 50,
               maximize = TRUE) # IMPROVE parameters later. Maximise AUC

(auc_best <- max(xgcv$evaluation_log[,"test_auc_mean"])) # Best AUC
(nrounds_best <- which.max(unlist(xgcv$evaluation_log[,"test_auc_mean"]))) # Step at which this was achieved

plot(test_auc_mean ~ iter, data = xgcv$evaluation_log, col = "darkgreen", 
     type = "b", pch = 16, ylab = "Test AUC", xlab = "Iteration") +
  abline(v = nrounds_best, lty = 2)

# Fit final model with best parameters
xgbmod <- xgb.train(data = xgdata, nrounds = nrounds_best, params = params)


# Determine predictive accuracy
Xte <- model.matrix(death ~ . - 1, datte)
Yte <- datte$death
probxgb <- predict(xgbmod, Xte)
predxgb <- factor(probxgb > 0.5, levels = c("FALSE", "TRUE"), labels = c("Survived", "Died")) # Label must match test labels

# Confusion Matrix
conf_matrix <- confusionMatrix(predxgb, as.factor(Yte))
print(conf_matrix)

# Accuracy
accuracy <- sum(predxgb == Yte) / length(Yte)
cat("Accuracy: ", accuracy, "\n")

# ROC Curve and AUC Calculation
roc_curve <- roc(Yte, probxgb)
plot(roc_curve, col = "blue", main = paste("ROC Curve (AUC =", round(auc(roc_curve), 2), ")"))


# Specificity and Sensitivity
sensitivity <- conf_matrix$byClass["Sensitivity"]
specificity <- conf_matrix$byClass["Specificity"]
cat("Sensitivity: ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")












# Use a more developed parameter method

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




# ------------------------------------
# Section 4: Comparison of models
# ------------------------------------

# We get test predictions from each model to compare
# Delete if unusued
# 1: Ridge predictions
ridge_prob <- predict(ridgemod, newx = Xte, type = "response")
ridge_pred <- factor(ridge_prob > .5, levels = c("FALSE", "TRUE"), 
                    labels = c("Survived", "Died"))
summary(ridge_pred)
# 2: Lasso predictions
lasso_prob <- predict(lassomod, newx = Xte, type = "response")
lasso_pred <- factor(lasso_prob > .5, levels = c("FALSE", "TRUE"), 
                    labels = c("Survived", "Died"))
summary(lasso_pred)
# 3: Elastic net predictions
enet_prob <- predict(enetmod, newx = Xte, type = "prob")
enet_pred <- factor(lasso_prob > .5, levels = c("FALSE", "TRUE"), 
                     labels = c("Survived", "Died"))
summary(enet_pred)




# Explore and compare feature importance from each model