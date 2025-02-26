# ML assessment. Candidate number: 492841

# Set your working directory
setwd("~")


# ------------------------------------
# Structure of this script:
# - Import relevant libraries and datasets
# - Carry out exploratory data analysis
# - Create a test/train split on the data
# - Train each regularised regression model
# - Train each tree-based model
# - Evaluate and compare each model
# Runtime should be 5-10 minutes
# ------------------------------------


# Import relevant libraries
install.packages("mboost")
install.packages("GGally")
install.packages("vcd")
library(tidyverse)
library(dplyr)
library(ggplot2)
library(glmnet)
library(mboost)
library(xgboost)
library(caret) # used for training, predicting, and pre-processing
library(pROC) # Required for the ROC curve
library(ggcorrplot)
library(GGally)
library(gridExtra)
library(forcats) # For ordering bar plots
library(reshape2)
library(vcd)  # For assocstats() to compute Cramér's V
library(patchwork)  # For combining plots

# Read in the dataset. Empty rows are ommitted by default
dat <- read.csv("assignment2025.csv")

# In case empty rows are not omitted
na.omit(dat)

# View the first 5 rows
head(dat, 5)

# Check for missing values
sum(is.na(dat)) # No missing values

# Convert categorical variables to factors
# This is required for proper handling in machine learning models
# This includes when converting to model matrix and determining outcome classes
categorical_cols <- names(dat)[sapply(dat, function(x) is.character(x) | is.factor(x))]
categorical_cols <- setdiff(categorical_cols, 'death')
dat[categorical_cols] <- lapply(dat[categorical_cols], factor)

# Convert the outcome to a factor for proper handling
dat$death <- factor(dat$death, levels = c(0, 1), labels = c("Survived", "Died"))

# ------------------------------------
# Section 1: Exploratory data analysis
# ------------------------------------

# Summarise each variable
summary(dat)

# Explore death rate by stroke subtype
death_rate <- dat %>%
  group_by(subtype) %>%
  summarise(drate = mean(as.numeric(death)-1) * 100)

# We see significant variation in death rate by subtype
# This suggests type, especially presence of TACS, could help predict death
ggplot(death_rate, aes(x = fct_reorder(subtype, drate), y = drate)) +
  geom_bar(stat = "identity", fill = "steelblue", color = "black") +
  geom_hline(yintercept = 4.7, linetype = "dashed", color = "black", size = 1) +
  labs(x = "Stroke Subtype", y = "Death Rate (%)", title = "Death Rate by Stroke Subtype") +
  theme_minimal() +
  theme_bw()


# Calculate percentage of cases who fall under each outcome (died or survived)
death_summary <- dat %>%
  count(death) %>%
  mutate(percent = 100 * n / sum(n))  # Convert count to percentage

# Outcome plot with percentages - clearly the majority (95.3%) survived
# This also shows a clear imbalance in the dataset, which will be addressed later
# We adjust these plots for suitable spacing, removing labels where appropriate and using a consistent colour pallette
# Colours for numerical boxplots
fill_colors <- c("Died" = "salmon", "Survived" = "lightblue")
ggplot(death_summary, aes(x = death, y = n, fill = death)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3) +  
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            vjust = -0.3, size = 5, color = "black") +  
  scale_fill_manual(values = fill_colors) +  
  labs(title = "Death Outcome Distribution", y = "Count", x = NULL) +  
  theme_bw() +
  theme(plot.margin = margin(10, 10, 20, 10), legend.position = "none") +  
  coord_cartesian(ylim = c(0, max(death_summary$n) * 1.03))

# Function to create boxplots with shared theme
boxplot_theme <- theme_bw() +
  theme(
    legend.position = "none",
    plot.margin = margin(3, 3, 3, 3),  
    axis.title.x = element_blank(),
    axis.title.y = element_blank()
  )

# Delay distribution by outcome
p1 <- ggplot(dat, aes(x = death, y = delay, fill = death)) +
  geom_boxplot(outlier.shape = 16, outlier.size = 1, color = "black") +
  scale_fill_manual(values = fill_colors) +
  labs(title = "Delay (hours)", y = "Delay Time") +
  boxplot_theme

# Age distribution by outcome
p2 <- ggplot(dat, aes(x = death, y = age, fill = death)) +
  geom_boxplot(outlier.shape = 16, outlier.size = 1, color = "black") +
  scale_fill_manual(values = fill_colors) +
  labs(title = "Age (years)", y = "Age") +
  boxplot_theme

# SBP distribution by outcome
p3 <- ggplot(dat, aes(x = death, y = sbp, fill = death)) +
  geom_boxplot(outlier.shape = 16, outlier.size = 1, color = "black") +
  scale_fill_manual(values = fill_colors) +
  labs(title = "Systolic BP", y = "Systolic BP") +
  boxplot_theme

# Arrange plots with reduced spacing
grid.arrange(
  arrangeGrob(p1, p2, p3, ncol = 3),
  ncol = 2,
  widths = c(4, 0.8)  # Reduce width ratio to tighten space
)


# Use as stacked bar plot to compare death rate by symptom
# Retrieve symptom variables in a concise manner
symptom_vars <- grep("^symptom", names(dat), value = TRUE)
symptom_death <- dat %>%
  pivot_longer(cols = all_of(symptom_vars), names_to = "Symptom", values_to = "Present") %>%
  filter(Present == "Y") %>%
  mutate(Symptom = gsub("^symptom", "", Symptom)) %>% # Remove symptom from the variable name
  group_by(Symptom) %>%
  summarise(DeathRate = mean(death == "Died") * 100) # Calculate the death rate per symptom

ggplot(symptom_death, aes(x = reorder(Symptom, DeathRate), y = DeathRate, fill = DeathRate)) +
  geom_col() +
  coord_flip() +
  labs(title = "Death Rate by Symptom", y = "Death Rate (%)", x = "Symptom") +
    scale_fill_gradient(guide = "none", low = "#6495ED", high = "#D22B2B") +  
  geom_hline(yintercept = 4.7, linetype = "dashed", color = "black", size = 1) +  # Dashed vertical line at 4.7% (baseline death rate)
  theme_bw() +
  theme(plot.margin = margin(10, 10, 20, 10))



# Function to compute Cramér's V
cramers_v <- function(x, y) {
  tbl <- table(x, y)
  stat <- suppressWarnings(assocstats(tbl))  # Ignore warnings for small tables
  return(stat$cramer)
}

# Retrieve all categorical columns, including death this time
categorical_cols <- names(dat)[sapply(dat, function(x) is.character(x) | is.factor(x))]
# Create an empty correlation matrix
n <- length(categorical_cols)
cat_corr_matrix <- matrix(NA, nrow = n, ncol = n, dimnames = list(categorical_cols, categorical_cols))

# Compute pairwise Cramér's V - this handles categorical variables, even if the number of levels differs
for (i in seq_along(categorical_cols)) {
  for (j in seq_along(categorical_cols)) {
    if (i == j) {
      cat_corr_matrix[i, j] <- 1  # Perfect correlation on the diagonal
    } else {
      cat_corr_matrix[i, j] <- cramers_v(dat[[categorical_cols[i]]], dat[[categorical_cols[j]]])
    }
  }
}

# Convert matrix to long format for ggplot
corr_melted <- melt(cat_corr_matrix)

# Create square heatmap
ggplot(corr_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +  
  scale_fill_gradient2(low = "#6495ED", mid = "white", high = "#D22B2B", midpoint = 0.5, name = "Cramér's V") +
  labs(title = "Categorical Variable Correlation Heatmap", x = "", y = "") +
  theme_bw() +  
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for readibility
    plot.margin = margin(10, 10, 20, 10),  
    panel.grid = element_blank()  
  ) +
  coord_fixed()  # Ensures a perfect square layout



# Compute death rate for treatment 1 types
death_treat1 <- dat %>%
  group_by(treat1) %>%
  summarise(drate = mean(death == "Died") * 100)

# Compute death rate for treatment 2 types
death_treat2 <- dat %>%
  group_by(treat2) %>%
  summarise(drate = mean(death == "Died") * 100)

# Create bar plot for treatment 1
plot1 <- ggplot(death_treat1, aes(x = treat1, y = drate, fill = treat1)) +
  geom_bar(stat = "identity", color = "black") +
  geom_hline(yintercept = 4.7, linetype = "dashed", color = "black", size = 1) +
  labs(title = "Death Rate by Treatment 1", x = "Treatment 1", y = "Death Rate (%)") +
  scale_fill_manual(values = c("L" = "#6495ED", "M" = "#FFA07A", "N" = "#D22B2B")) +  
  theme_bw() +
  theme(legend.position = "none")

# Create bar plot for treatment 2
plot2 <- ggplot(death_treat2, aes(x = treat2, y = drate, fill = treat2)) +
  geom_bar(stat = "identity", color = "black") +
  geom_hline(yintercept = 4.7, linetype = "dashed", color = "black", size = 1) +
  labs(title = "Death Rate by Treatment 2", x = "Treatment 2", y = "Death Rate (%)") +
  scale_fill_manual(values = c("Y" = "#6495ED", "N" = "#D22B2B")) +  
  theme_bw() +
  theme(legend.position = "none")

# Place the two plots side-by-side
plot1 + plot2



# ------------------------------------
# Split into training + validation sets
# ------------------------------------

# We use createDataPartition to preserve the death distribution, reducing bias
psamp <- .2 # Use an 80-20 train-test split
set.seed(22) # For reproducibility
testind <- createDataPartition(y = dat$death, p = psamp)[[1]]

# Split data into testing and training
# We will use dattr to train each model below
dattr <- dat[-testind,]
datte <- dat[testind,]


# Note that we will not oversample on the validation dataset
# This is to preserve the underlying low death rate of ~5%, asserting generalisability

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

# Obtain the best lambda, which minimises log-loss
ridgemod$lambda.min

# Train a lasso model with alpha = 1 (default)
lassomod <- cv.glmnet(Xtr, Ytr,
                      family = "binomial")
plot(lassomod, main = "Lasso tuning")  
abline(v = log(lassomod$lambda.min), col = "blue", lty = 2)  
abline(v = log(lassomod$lambda.1se), col = "red", lty = 2)  
text(log(lassomod$lambda.min), max(lassomod$cvm), "Min log-loss", pos = 4, col = "blue") # Caption each highlighted value
text(log(lassomod$lambda.1se), max(lassomod$cvm), "1se", pos = 4, col = "red")  


# Obtain the best lambda, and most regularised one
(lassomod$lambda.min)
(lassomod$lambda.1se)

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

# After constructing the previous models, we will adjust the elastic net model
# This is done through adjusting the death balance of the dataset
# We also centre and scale continuous variables to avoid bias based on variable ranges
set.seed(42) # For reproducibility
dattr_balanced <- upSample(x = dattr[, -which(names(dattr) == "death")], y = dattr$death) %>%
  rename(death = Class)

# Explore balance
summary(dattr_balanced$death) # Same number in each group

# Select if downsampling is used instead
dattr_down <- downSample(x = dattr[, -which(names(dattr) == "death")], y = dattr$death) %>%
  rename(death = Class)

# Explore balance
summary(dattr_down$death) # Same number in each group

#### Plot the death distribution for the original and upsampled data

# Calculate death percentages for original training set
death_summary_tr <- dattr %>%
  count(death) %>%
  mutate(percent = 100 * n / sum(n))

# Calculate death percentages for upsampled training set
death_summary_bal <- dattr_balanced %>%
  count(death) %>%
  mutate(percent = 100 * n / sum(n))

# Plot for original training data
p1 <- ggplot(death_summary_tr, aes(x = death, y = n, fill = death)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3) +
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            vjust = -0.3, size = 5, color = "black") +
  scale_fill_manual(values = c("Died" = "salmon", "Survived" = "lightblue")) +
  labs(title = "Death Outcome Distribution (Original)", y = "Count", x = NULL) +
  theme_bw() +
  theme(plot.margin = margin(10, 10, 20, 10), legend.position = "none") +
  coord_cartesian(ylim = c(0, max(death_summary_tr$n) * 1.03))

# Plot for upsampled training data
p2 <- ggplot(death_summary_bal, aes(x = death, y = n, fill = death)) +
  geom_bar(stat = "identity", color = "black", linewidth = 0.3) +
  geom_text(aes(label = paste0(round(percent, 1), "%")), 
            vjust = -0.3, size = 5, color = "black") +
  scale_fill_manual(values = c("Died" = "salmon", "Survived" = "lightblue")) +
  labs(title = "Death Outcome Distribution (Upsampled)", y = "Count", x = NULL) +
  theme_bw() +
  theme(plot.margin = margin(10, 10, 20, 10), legend.position = "none") +
  coord_cartesian(ylim = c(0, max(death_summary_bal$n) * 1.03))

# Compare both plots now
grid.arrange(p1, p2, ncol = 2) # Clearly there is more balance with the upsampled data

###

# Scale and center continuous variables
# This ensures variables with greater range don't disproportionately shrink coefficients
# Centering prevents bias towards certain variables due to their scale
continuous_vars <- c("age", "delay", "sbp")
preProc <- preProcess(dattr_balanced[, continuous_vars], method = c("center", "scale"))
dattr_balanced[, continuous_vars] <- predict(preProc, dattr_balanced[, continuous_vars])
# Downsampling equivalent
preProc_down <- preProcess(dattr_down[, continuous_vars], method = c("center", "scale"))
dattr_down[, continuous_vars] <- predict(preProc, dattr_down[, continuous_vars])

# Train the adjusted elastic net model - use dattr_balanced or dattr_down as appropriate
# We find better results with upsampled data due to more data to train the model on
# However, the duplicates can also lead to bias and less variance in the training data
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

# NOTE: we used upSample here, leading to large counts of duplicated data
# Alternatively, we could use SMOTE to create synthetic data reflecting the minority class
# This maintains variance in the dataset, however is more computationally expensive

# ------------------------------------
# Section 3: Tree-based methods
# ------------------------------------

# Whilst Random Forests are commonly used, XGBoost is generally more powerful
# This involves better handling of imbalance, alongside boosting for more optimal performance
# Random Forest works better with downsampling than upsampling to reduce bias
# In this case, downsampling returns much fewer data points, reducing predictive ability

# XGBoost model

# Load testing and training matrices in a supported format
Xtr_mat <- model.matrix(death ~ . -1, dattr)
Xte_mat <- model.matrix(death ~ . -1, datte)
Ytr_num <- as.numeric(dattr$death) - 1
Yte_num <- as.numeric(datte$death) - 1

# Convert to DMatrix for XGBoost compatibility
xgtrain <- xgb.DMatrix(data = Xtr_mat, label = Ytr_num)

# Create the basic XGBoost model using cross-validation
# In this context, "test" does not mean our datte dataset from before
# Rather, it means the average results on the validation set within each fold
params <- list(objective = "binary:logistic", eval_metric = "auc")
xgcv <- xgb.cv(data = xgtrain, nrounds = 500, nfold = 10, 
                      verbose = FALSE, params = params)

# Determine best AUC and the number of rounds at which this was achieved
(auc_best <- max(xgcv$evaluation_log[,"test_auc_mean"])) # Best AUC
(nrounds_best <- which.max(unlist(xgcv$evaluation_log[,"test_auc_mean"]))) # Step at which this was achieved

# Plot training vs testing AUC
ggplot(xgcv$evaluation_log, aes(x = iter)) +
  geom_line(aes(y = train_auc_mean), color = "blue", size = 1) +
  geom_line(aes(y = test_auc_mean), color = "darkgreen", size = 1) +
  labs(title = "Training vs Validation AUC", x = "Iteration", y = "AUC") +
  theme_bw() +
  theme(plot.margin = margin(10, 10, 20, 10))

# Plot AUC by iteration
ggplot(xgcv$evaluation_log, aes(x = iter, y = test_auc_mean)) +
  geom_line(color = "darkgreen", size = 1) +  # Line plot for AUC
  geom_point(color = "darkgreen", size = 1) +  # Points on the line
  geom_vline(xintercept = nrounds_best, linetype = "dashed", color = "red") +  
  labs(title = "AUC by Iteration", x = "Iteration", y = "Validation AUC") +  
  theme_bw() +  # Consistent theme
  theme(plot.margin = margin(10, 10, 20, 10))  

# Fit final basic model with best parameters
xgbmod <- xgb.train(data = xgtrain, nrounds = nrounds_best, params = params)

# Generate confusion matrix
# IMPORTANT: this code takes the first factor, Survived, as the positive class
# Therefore the sensitivity and specificity returned here should be reversed
# In the table further below, I simply relabel these columns to account for this
# The alternative option is to refactor our known labels in Ytr and Yte
probs <- predict(xgbmod, Xte)
roc_curve <- roc(Yte, probs)
best_index <- which.max(roc_curve$sensitivities + roc_curve$specificities - 1)
xgb_thd <- roc_curve$thresholds[best_index]
pred_labels <- factor(ifelse(probs > xgb_thd, "Died", "Survived"), 
                      levels = c("Survived", "Died"))
confusion <- confusionMatrix(pred_labels, Yte)

# Convert confusion matrix to a data frame
confusion_table <- as.data.frame(as.table(confusion$table))

# Ensure the factor levels are in the correct order
confusion_table$Prediction <- factor(confusion_table$Prediction, levels = c("Survived", "Died"))
confusion_table$Reference <- factor(confusion_table$Reference, levels = c("Survived", "Died"))

# Plot confusion matrix
ggplot(confusion_table, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "salmon") +  # Light blue to salmon gradient
  geom_text(aes(label = Freq), color = "black", size = 5) +  # Add frequency text in each cell
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_bw() +
  theme(plot.margin = margin(10, 10, 20, 10)) +
  scale_y_discrete(limits = rev(levels(confusion_table$Reference)))  # Flip the y-axis



# Create a more advanced XGBoost model with extra parameters and balanced data
Xtr_bal_mat <- model.matrix(death ~ . -1, dattr_balanced)
Ytr_bal_num <- as.numeric(dattr_balanced$death) - 1
xgtrain_bal <- xgb.DMatrix(data = Xtr_bal_mat, label = Ytr_bal_num)

# Define advanced parameters
params_adv <- list(
  objective = "binary:logistic",  # Probability estimation for binary classification
  eval_metric = "auc",            # AUC as evaluation metric - log-loss may be better for probability estimation
  eta = 0.05,                     # Learning rate
  max_depth = 6,                  # Tree depth
  min_child_weight = 8,           # Minimum leaf weight
  subsample = 0.8,                # Prevent overfitting
  colsample_bytree = 0.8,         # Feature sampling for diversity
  scale_pos_weight = sum(Ytr_bal_num == 0) / sum(Ytr_bal_num == 1),  # Handle class imbalance
  lambda = 1.5,                   # L2 regularization
  alpha = 0.5,                    # L1 regularization
  gamma = 2                       # Minimum loss reduction to make a split
)

# Train XGBoost model using 10-fold cross-validation
set.seed(123)
xgcv_adv <- xgb.cv(
  params = params_adv,
  data = xgtrain_bal,
  nrounds = 200,                  # Maximum number of boosting rounds
  nfold = 10,                     # Number of folds for cross-validation
  early_stopping_rounds = 20,     # Stop if no improvement
  verbose = FALSE                 # Set to TRUE to see result of each iteration
)

# Extract the best AUC from cross-validation and optimal rounds
(auc_best_adv <- max(xgcv_adv$evaluation_log[,"test_auc_mean"]))  # Best AUC
(nrounds_best_adv <- which.max(unlist(xgcv_adv$evaluation_log[,"test_auc_mean"])))  # Step at which best AUC was achieved

# Plot AUC by iteration for the advanced model
ggplot(xgcv_adv$evaluation_log, aes(x = iter, y = test_auc_mean)) +
  geom_line(color = "darkgreen", size = 1) +  # Line plot for AUC
  geom_point(color = "darkgreen", size = 2) +  # Points on the line
  geom_vline(xintercept = nrounds_best_adv, linetype = "dashed", color = "red") +  
  labs(title = "AUC by Iteration", x = "Iteration", y = "Validation AUC") +  
  theme_bw() +  # Consistent theme
  theme(plot.margin = margin(10, 10, 20, 10))  # Adjust margin for better spacing

# Training vs Validation AUC for the advanced model
ggplot(xgcv_adv$evaluation_log, aes(x = iter)) +
  geom_line(aes(y = train_auc_mean), color = "blue", size = 1) +  # Training AUC
  geom_line(aes(y = test_auc_mean), color = "darkgreen", size = 1) +  # Validation AUC
  labs(title = "Training vs Validation AUC", x = "Iteration", y = "AUC") +  
  theme_bw() +  # Consistent theme
  theme(plot.margin = margin(10, 10, 20, 10))  


# Convert the balanced training data to DMatrix format
Xtrain_adv <- model.matrix(death ~ . - 1, dattr_balanced)
Ytrain_adv <- factor(dattr_balanced$death, levels = c("Survived", "Died"), labels = c("Survived", "Died"))
xgb_train_adv <- xgb.DMatrix(data = Xtrain_adv, label = as.numeric(Ytrain_adv) - 1)

# Train the final model using the best nrounds and parameters from xgcv_adv
final_model_adv <- xgb.train(
  params = params_adv,
  data = xgb_train_adv,
  nrounds = nrounds_best_adv,         
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
selected_features_adj <- rownames(enet_coef_adj)[-1]  # Make predictions using only retained features

# Ensure predictions are made on scaled data for compatibility
Xtr_scaled <- scale(Xtr)
Xte_scaled <- scale(Xte, center = attr(Xtr_scaled, "scaled:center"), 
                    scale = attr(Xtr_scaled, "scaled:scale"))
Xte_subset_adj <- Xte_scaled[, selected_features_adj]
enet_prob_adj <- predict(enet_fin_adj, newx = Xte_subset_adj, type = "response", s = best_lambda_adj)
enet_pred_adj <- factor(enet_prob_adj > .5, levels = c("FALSE", "TRUE"), labels = c("Survived", "Died"))

# 5: Simple XGBoost predictions
xgb_prob <- predict(xgbmod, Xte)
xgb_pred <- factor(xgb_prob > 0.5, levels = c("FALSE", "TRUE"), labels = c("Survived", "Died")) # Label must match test labels

# 6: Enhanced XGBoost predictions 
xgb_prob_adv <- predict(final_model_adv, Xte)
xgb_pred_adv <- factor(xgb_prob_adv > 0.5, levels = c("FALSE", "TRUE"), labels = c("Survived", "Died"))


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
  
  # Calculate balanced accuracy
  balanced_accuracy <- (cm$byClass["Sensitivity"] + cm$byClass["Specificity"]) / 2
  
  # Extract relevant metrics
  results <- list(
    model = mod_name,
    auc = auc_score,
    sensitivity = cm$byClass["Sensitivity"],  # Recall for "Died"
    specificity = cm$byClass["Specificity"],  # True negative rate for "Survived"
    f1_score = f1_score,
    bal_acc = balanced_accuracy, # Used over accuracy due to imbalance
    threshold = best_threshold # Include threshold used
  )
  
  return(results)
}

# Evaluate selected models
ridge_eval <- evaluate_model("Ridge", ridge_prob, Yte)
lasso_eval <- evaluate_model("Lasso", lasso_prob, Yte)
enet_eval <- evaluate_model("Elastic Net", enet_prob, Yte)
enet_eval_adj <- evaluate_model("Adjusted Elastic Net", enet_prob_adj, Yte)
xgb_eval <- evaluate_model("XGBoost", xgb_prob, Yte)
xgb_eval_adv <- evaluate_model("Advanced XGBoost", xgb_prob_adv, Yte)

# Print table of results
# IMPORTANT: confusion matrix took Survived as the positive class
# Therefore, I have switched the labels of sensitivity and specificity
results_df <- data.frame(
  Model = c("Ridge", "Lasso", "Elastic Net", "Adjusted Elastic Net", "XGBoost", "Advanced XGBoost"),
  AUC = round(c(ridge_eval$auc, lasso_eval$auc, enet_eval$auc, enet_eval_adj$auc, xgb_eval$auc, xgb_eval_adv$auc), 3),
  Specificity = round(c(ridge_eval$sensitivity, lasso_eval$sensitivity, enet_eval$sensitivity, enet_eval_adj$sensitivity, xgb_eval$sensitivity, xgb_eval_adv$sensitivity), 3),
  Sensitivity = round(c(ridge_eval$specificity, lasso_eval$specificity, enet_eval$specificity, enet_eval_adj$specificity, xgb_eval$specificity, xgb_eval_adv$specificity), 3),
  F1 = round(c(ridge_eval$f1_score, lasso_eval$f1_score, enet_eval$f1_score, enet_eval_adj$f1_score, xgb_eval$f1_score, xgb_eval_adv$f1_score), 3),
  Balanced_Acc = paste0(round(c(ridge_eval$bal_acc, lasso_eval$bal_acc, enet_eval$bal_acc, enet_eval_adj$bal_acc, xgb_eval$bal_acc, xgb_eval_adv$bal_acc) *100, 2), "%"),
  Threshold = round(c(ridge_eval$threshold, lasso_eval$threshold, enet_eval$threshold, enet_eval_adj$threshold, xgb_eval$threshold, xgb_eval_adv$threshold), 3)
)
# Rename the balanced accuracy column to look nicer
colnames(results_df)[colnames(results_df) == "Balanced_Acc"] <- "Balanced Accuracy"

# View the evaluation dataframe for each model
print(results_df)
View(results_df)


# Produce the ROC curve for each model
plot_roc_curve <- function(model_name, probs, true_vals) {
  true_numeric <- as.numeric(true_vals == "Died")
  roc_curve <- roc(true_numeric, probs)
  auc_score <- auc(roc_curve)
  
  plot(roc_curve, main = paste("ROC Curve -", model_name), col = "blue", lwd = 2)
  
  text(x = 1, y = 0.95, labels = paste("AUC =", round(auc_score, 3)), adj = 0, cex = 1.2, font = 2)
}

# Plot ROC curves for each model
plot_roc_curve("Ridge", ridge_prob, Yte)
plot_roc_curve("Lasso", lasso_prob, Yte)
plot_roc_curve("Elastic Net", enet_prob, Yte)
plot_roc_curve("Adjusted Elastic Net", enet_prob_adj, Yte)
plot_roc_curve("XGBoost", xgb_prob, Yte)
plot_roc_curve("Advanced XGBoost", xgb_prob_adv, Yte)

# ------------------------------------
# Comparing feature importances
# ------------------------------------

# Explore and compare feature importance from each model
# 1. Ridge
ridge_imp <- coef(ridgemod)[-1,]  # Removing the intercept as this doesn't relate to any specific feature
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

# 5. Simple XGBoost - Feature Importance
xgb_imp <- xgb.importance(model = xgbmod)

# 6. Enhanced XGBoost - Feature Importance
xgb_adv_imp <- xgb.importance(model = final_model_adv)

# Combine all models feature importances into a data frame
ridge_top10 <- sort(ridge_imp, decreasing = TRUE)[1:10]
lasso_top10 <- sort(lasso_imp, decreasing = TRUE)[1:10]
enet_top10 <- sort(enet_coef, decreasing = TRUE)[1:10]
enet_adj_top10 <- sort(enet_coef_adj, decreasing = TRUE)[1:10]
xgb_top10 <- xgb_imp[1:10, ]
xgb_adv_top10 <- xgb_adv_imp[1:10, ]

# Prepare data for plotting
ridge_df <- data.frame(Feature = names(ridge_top10), Importance = ridge_top10)
lasso_df <- data.frame(Feature = names(lasso_top10), Importance = lasso_top10)
enet_df <- data.frame(Feature = names(enet_top10), Importance = enet_top10)
enet_adj_df <- data.frame(Feature = names(enet_adj_top10), Importance = enet_adj_top10)

xgb_df_gain <- data.frame(Feature = xgb_top10$Feature, Importance = xgb_top10$Gain)
xgb_df_freq <- data.frame(Feature = xgb_top10$Feature, Importance = xgb_top10$Frequency)

xgb_adv_df_gain <- data.frame(Feature = xgb_adv_top10$Feature, Importance = xgb_adv_top10$Gain)
xgb_adv_df_freq <- data.frame(Feature = xgb_adv_top10$Feature, Importance = xgb_adv_top10$Frequency)

# Ridge plot
ggplot(ridge_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#A3D9A5", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 Ridge Feature Importances", x = "Feature", y = "Importance (abs)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  # Adjusted margin for spacing

# Lasso plot
ggplot(lasso_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#A3B9D7", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 Lasso Feature Importances", x = "Feature", y = "Importance (abs)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  # Adjusted margin for spacing

# Elastic Net plot
ggplot(enet_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#F7C57F", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 Elastic Net Feature Importances", x = "Feature", y = "Importance (abs)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  

# Adjusted Elastic Net plot
ggplot(enet_adj_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#F7C57F", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 Adjusted Elastic Net Feature Importances", x = "Feature", y = "Importance (abs)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  

# Simple XGBoost Gain plot
ggplot(xgb_df_gain, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#D97D7D", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 XGBoost Feature Importances", x = "Feature", y = "Importance (Gain)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  

# Simple XGBoost Frequency plot
ggplot(xgb_df_freq, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#D8A1D6", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 XGBoost Feature Importances", x = "Feature", y = "Importance (Frequency)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  

# Advanced XGBoost Gain plot
ggplot(xgb_adv_df_gain, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#F6A5A5", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 Advanced XGBoost Feature Importances (Gain)", x = "Feature", y = "Importance (Gain)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  

# Advanced XGBoost Frequency plot
ggplot(xgb_adv_df_freq, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#F6A5A5", color = "black", linewidth = 0.3) +  
  coord_flip() +
  labs(title = "Top 10 Advanced XGBoost Feature Importances (Frequency)", x = "Feature", y = "Importance (Frequency)") +
  theme_bw() + 
  theme(plot.margin = margin(10, 10, 20, 10))  


# Enter the top 10 features for each model into a table
list(
  Ridge_Top_10 = ridge_df,
  Lasso_Top_10 = lasso_df,
  Elastic_Net_Top_10 = enet_df,
  Elastic_Net_Adjusted_Top_10 = enet_adj_df,
  XGBoost_Top_10_Gain = xgb_df_gain,
  XGBoost_Top_10_Frequency = xgb_df_freq,
  Advanced_XGBoost_Top_10_Gain = xgb_adv_df_gain,
  Advanced_XGBoost_Top_10_Frequency = xgb_adv_df_freq
)

