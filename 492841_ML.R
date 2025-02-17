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
library(caret) # used for training, predicting, and pre-processing
library(pROC)
library(ggcorrplot)
install.packages("GGally") # remove if unused
library(GGally)
library(gridExtra)

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

# Convert the outcome to a factor for proper handling
dat$death <- factor(dat$death, levels = c(0, 1), labels = c(0, 1))


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

grid.arrange(p1, p2, p3, ncol = 2)

# 4. Bar plots for categorical variables by death outcome
categorical_vars <- setdiff(categorical_cols, "death")

p_list_cat <- lapply(categorical_vars, function(var) {
  ggplot(dat, aes(x = .data[[var]], fill = death)) +
    geom_bar(position = "fill") +
    labs(title = paste("Proportion of", var, "by Death Outcome"), y = "Proportion")
})
do.call(grid.arrange, c(p_list_cat, ncol = 2))

# 5. Correlation plot for numeric variables
numeric_cols <- c("delay", "age", "sbp")
num_vars <- dat %>% select(all_of(numeric_cols))
cor_matrix <- cor(num_vars, use = "complete.obs")
ggcorrplot(cor_matrix, lab = TRUE, title = "Correlation Plot of Numeric Variables")

# 6. Pair plot for key numerical variables
ggpairs(dat, columns = which(names(dat) %in% c("death", numeric_cols)))

# 7. Stacked bar plot for symptom variables by death outcome
symptom_vars <- grep("symptom", names(dat), value = TRUE)
dat_long <- dat %>% pivot_longer(cols = all_of(symptom_vars), names_to = "Symptom", values_to = "Present")

ggplot(dat_long, aes(x = Symptom, fill = death)) +
  geom_bar(position = "fill") +
  labs(title = "Proportion of Symptoms by Death Outcome", y = "Proportion") +
  coord_flip()

# ------------------------------------
# Split into training + validation
# ------------------------------------

# we use createDataPartition to preserve the death distribution, reducing bias

psamp <- .2 # use an 80-20 train-test split
set.seed(22) # for reproducibility
testind <- createDataPartition(y = dat$death, p = psamp)[[1]]

# split data
dattr <- dat[-testind,]
datte <- dat[testind,]


# note that we will not oversample on the validation dataset
# this is to preserve the underlying low death rate of ~5%


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





# XGBoost

# Extract data as matrices
#X <- model.matrix(death ~ . - 1, dattr)
#Y <- dattr$death

# Try upsampling
up_data <- upSample(x = dattr %>% select(-death), y = dattr$death) %>%
  rename(death = Class)

X <- model.matrix(death ~ . - 1, up_data)
Y <- factor(up_data$death, levels = c(0, 1), labels = c(0, 1))

# Convert to DMatrix - UPDATE NAME
xgdata <- xgb.DMatrix(data = X, label = as.numeric(Y) - 1)

# Tune the number of iterations
set.seed(55)
params <- list(objective = "binary:logistic", eval_metric = "auc",
               scale_pos_weight = 10, # pay more attention to minority class to improve specificity
               max_depth = 6,
               min_child_weight = 8,
               subsample = 0.8,
               colsample_bytree = 0.8,
               eta = 0.05, # control the learning rate
               gamma = 2, # reduction required before a split is made
               lambda = 1.5,
               alpha = 0.95) # controls regularisation
xgcv <- xgb.cv(data = xgdata, nrounds = 300, nfold = 6, verbose = F, params = params,
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
predxgb <- factor(probxgb > 0.5, levels = c("FALSE", "TRUE"), labels = c(0, 1))

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