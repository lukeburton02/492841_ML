# ML assessment. Candidate number: 492841

# Set your working directory
setwd("C:/Users/lukeb/Downloads/LSHTM/TERM 2/Machine Learning/ML assessment/492841_ML")

# Import relevant libraries
install.packages("mboost")
install.packages("GGally") # remove if unused
install.packages("smotefamily") # used for oversampling
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
library(smotefamily) # for balancing training data


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



# Train an elastic net model with alpha in [0,1]
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

# Obtain the best tuning parameters
print(enetmod$bestTune)

# Plot the ROC for each model
plot(enetmod, xTrans = log10)


# We could minimise logLoss instead, although the differences are marginal here
# enetmod <- train(death ~ .,
#                 data = dattr,
#                 method = "glmnet", 
#                 family = "binomial",
#                 tuneGrid = expand.grid(alpha = 0:10 / 10, 
#                                        lambda = 10^seq(-5, 2, length.out = 100)),
#                 trControl = trainControl("cv",
#                                          number = 10,
#                                          summaryFunction = mnLogLoss,
#                                          classProbs = TRUE),
#                 metric = "logLoss")

# After constructing the previous models, we will enhance the elastic net model
# This is done through adjusting the death balance of the dataset
# We also centre and scale continuous variables to avoid bias based on ranges
set.seed(42)
dattr_balanced <- upSample(x = dattr[, -which(names(dattr) == "death")], y = dattr$death) %>%
  rename(death = Class)

# Explore balance
summary(dattr_balanced$death)

# Select if downsampling is used instead
dattr_down <- downSample(x = dattr[, -which(names(dattr) == "death")], y = dattr$death) %>%
  rename(death = Class)

# Explore balance
summary(dattr_down$death)

# Scale and center continuous variables
continuous_vars <- c("age", "delay", "sbp")
preProc <- preProcess(dattr_balanced[, continuous_vars], method = c("center", "scale"))
dattr_balanced[, continuous_vars] <- predict(preProc, dattr_balanced[, continuous_vars])
# Downsampling equivalent
preProc_down <- preProcess(dattr_down[, continuous_vars], method = c("center", "scale"))
dattr_down[, continuous_vars] <- predict(preProc, dattr_down[, continuous_vars])

# Train the adjusted elastic net model - use dattr_balanced or dattr_down as appropriate
enetmod_adj <- train(death ~ .,
                          data = dattr_balanced,
                          method = "glmnet",
                          family = "binomial",
                          tuneGrid = expand.grid(alpha = 0:10 / 10, 
                                                 lambda = 10^seq(-5, 2, length.out = 100)),
                          trControl = trainControl("cv", number = 10, 
                                                   summaryFunction = twoClassSummary, 
                                                   classProbs = TRUE),
                          metric = "ROC")


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
# 3: Elastic net predictions on best model
best_lambda <- enetmod$bestTune$lambda
enet_fin <- enetmod$finalModel

enet_coef <- coef(enet_fin, s = best_lambda)
enet_coef <- as.matrix(enet_coef)

selected_features <- rownames(enet_coef)[-1] # Use only applicable features for prediction
Xte_subset <- Xte[, selected_features]
enet_prob <- predict(enet_fin, newx = Xte_subset, type = "response", s = best_lambda)
enet_pred <- factor(enet_prob > .5, levels = c("FALSE", "TRUE"), 
                     labels = c("Survived", "Died"))
summary(enet_pred)
# 4: Elastic net predictions using more balanced and scaled dataset
best_lambda_adj <- enetmod_adj$bestTune$lambda
enet_fin_adj <- enetmod_adj$finalModel

enet_coef_adj <- coef(enet_fin_adj, s = best_lambda_adj)
selected_features_adj <- rownames(enet_coef_adj)[-1]  # Apply only selected features

# Ensure predictions are made on scaled data for compatibility
Xtr_scaled <- scale(Xtr)
Xte_scaled <- scale(Xte, center = attr(Xtr_scaled, "scaled:center"), 
                    scale = attr(Xtr_scaled, "scaled:scale"))
Xte_subset_adj <- Xte_scaled[, selected_features_adj]
enet_prob_adj <- predict(enet_fin_adj, newx = Xte_subset_adj, type = "response", s = best_lambda_adj)
enet_pred_adj <- factor(enet_prob_adj > .5, levels = c("FALSE", "TRUE"), labels = c("Survived", "Died"))




# Evaluate the performance of a model, given predictions and true outcomes
evaluate_model <- function(mod_name, probs, true_vals) {
  
  # Convert true values to numeric (Died = 1, Survived = 0)
  true_numeric <- as.numeric(true_vals == "Died")
  
  # Convert probs to numeric vector for ROC compatibility
  probs <- as.vector(probs)
  
  # Compute ROC and AUC
  roc_curve <- roc(true_numeric, probs)
  auc_score <- auc(roc_curve)
  
  # Use Youden's J index to determine optimal threshold
  best_index <- which.max(roc_curve$sensitivities + roc_curve$specificities - 1)
  best_threshold <- roc_curve$thresholds[best_index]
  
  # Generate predictions using the above threshold
  pred_labels <- factor(ifelse(probs > best_threshold, "Died", "Survived"), 
                        levels = c("Survived", "Died"))
  
  # Compute confusion matrix
  cm <- confusionMatrix(pred_labels, true_vals)
  
  # Calculate precision and recall, for F1 score
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Sensitivity"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # Calculate Balanced Accuracy
  balanced_accuracy <- (cm$byClass["Sensitivity"] + cm$byClass["Specificity"]) / 2
  
  # Extract relevant metrics
  results <- list(
    model = mod_name,
    auc = auc_score,
    sensitivity = cm$byClass["Sensitivity"],  # Recall for "Died"
    specificity = cm$byClass["Specificity"],  # True negative rate for "Survived"
    f1_score = f1_score,
    balanced_accuracy = balanced_accuracy # Used over accuracy due to imbalance
  )
  
  return(results)
}

# Evaluate selected models
ridge_eval <- evaluate_model("Ridge", ridge_prob, Yte)
lasso_eval <- evaluate_model("Lasso", lasso_prob, Yte)
enet_eval <- evaluate_model("Elastic Net", enet_prob, Yte)
enet_eval_adj <- evaluate_model("Adjusted Elastic Net", enet_prob_adj, Yte)

# Print table of results
results_df <- data.frame(
  Model = c("Ridge", "Lasso", "Elastic Net", "Adjusted Elastic Net"),
  AUC = round(c(ridge_eval$auc, lasso_eval$auc, enet_eval$auc, enet_eval_adj$auc), 3),
  Sensitivity = round(c(ridge_eval$sensitivity, lasso_eval$sensitivity, enet_eval$sensitivity, enet_eval_adj$sensitivity), 3),
  Specificity = round(c(ridge_eval$specificity, lasso_eval$specificity, enet_eval$specificity, enet_eval_adj$specificity), 3),
  F1 = round(c(ridge_eval$f1_score, lasso_eval$f1_score, enet_eval$f1_score, enet_eval_adj$f1_score), 3),
  Balanced_Acc = paste0(round(c(ridge_eval$balanced_accuracy, lasso_eval$balanced_accuracy, enet_eval$balanced_accuracy, enet_eval_adj$balanced_accuracy) *100, 2), "%")
)
# Rename the balanced accuracy column to look nicer
colnames(results_df)[colnames(results_df) == "Balanced_Acc"] <- "Balanced Accuracy"

print(results_df)



# Explore and compare feature importance from each model
# 1. Ridge
ridge_imp <- coef(ridgemod)[-1,]  # Removing the intercept
ridge_imp <- abs(ridge_imp)  # Taking absolute value to measure importance

# 2. Lasso
lasso_imp <- coef(lassomod)[-1,]
lasso_imp <- abs(lasso_imp)

# 3. Elastic Net
enet_coef <- coef(enet_fin, s = best_lambda)  # Extracting coefficients at the best lambda value
enet_coef <- as.numeric(enet_coef[-1])
names(enet_coef) <- rownames(coef(enet_fin, s = best_lambda))[-1] 
enet_coef <- abs(enet_coef)

# 4. Adjusted Elastic Net
enet_coef_adj <- coef(enet_fin_adj, s = best_lambda_adj)  # Extracting coefficients at the best lambda value
enet_coef_adj <- as.numeric(enet_coef_adj[-1])
names(enet_coef_adj) <- rownames(coef(enet_fin_adj, s = best_lambda_adj))[-1] 
enet_coef_adj <- abs(enet_coef_adj)

# Combine all models feature importances into a data frame
ridge_top10 <- sort(ridge_imp, decreasing = TRUE)[1:10]
lasso_top10 <- sort(lasso_imp, decreasing = TRUE)[1:10]
enet_top10 <- sort(enet_coef, decreasing = TRUE)[1:10]
enet_adj_top10 <- sort(enet_coef_adj, decreasing = TRUE)[1:10]

# Prepare data for plotting
ridge_df <- data.frame(Feature = names(ridge_top10), Importance = ridge_top10)
lasso_df <- data.frame(Feature = names(lasso_top10), Importance = lasso_top10)
enet_df <- data.frame(Feature = names(enet_top10), Importance = enet_top10)
enet_adj_df <- data.frame(Feature = names(enet_adj_top10), Importance = enet_adj_top10)

# Ridge plot
ggplot(ridge_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "green") +
  coord_flip() +
  labs(title = "Top 10 Ridge Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal()

# Lasso plot
ggplot(lasso_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Lasso Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal()

# Elastic Net plot
ggplot(enet_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Top 10 Elastic Net Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal()

# Adjusted Elastic Net plot
ggplot(enet_adj_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Top 10 Adjusted Elastic Net Feature Importances", x = "Feature", y = "Importance") +
  theme_minimal()

# Print top 10 features for each model in a table
list(
  Ridge_Top_10 = ridge_df,
  Lasso_Top_10 = lasso_df,
  Elastic_Net_Top_10 = enet_df,
  Elastic_Net_Adjusted_Top_10 = enet_adj_df
)
