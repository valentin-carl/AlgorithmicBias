```R
# importing libraries
library(tidyverse)
library(randomForest)
library(ggthemes)
library(gridExtra)
library(cowplot)

# importing data
# source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
credit_data <- read_table2("data/german.data",
                           col_names = FALSE,
                           col_types = cols(
                             X1  = col_factor(levels = c("A11", "A12", "A13", "A14")),
                             X3  = col_factor(levels = c("A30", "A31", "A32", "A33", "A34")),
                             X4  = col_factor(levels = c("A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410")),
                             X6  = col_factor(levels = c("A61", "A62", "A63", "A64", "A65")),
                             X7  = col_factor(levels = c("A71", "A72", "A73", "A74", "A75")),
                             X9  = col_factor(levels = c("A91", "A92", "A93", "A94", "A95")),
                             X10 = col_factor(levels = c("A101", "A102", "A103")),
                             X12 = col_factor(levels = c("A121", "A122", "A123", "A124")),
                             X14 = col_factor(levels = c("A141", "A142", "A143")),
                             X15 = col_factor(levels = c("A151", "A152", "A153")),
                             X17 = col_factor(levels = c("A171", "A172", "A173", "A174")),
                             X19 = col_factor(levels = c("A191", "A192")),
                             X20 = col_factor(levels = c("A201", "A202")),
                             X21 = col_factor(levels = c("1", "2"))
                             )
                           )

names(credit_data) <- c("CheckingAccountStatus",
                        "CreditDuration",
                        "CreditHistory",
                        "Purpose",
                        "CreditAmount",
                        "SavingsAccount",
                        "EmploymentDuration",
                        "InstallmentRate",
                        "RelationshipStatusGender",
                        "OtherDebtorsGuarantors",
                        "DurationResidence",
                        "Property",
                        "Age",
                        "OtherInstallmentPlans",
                        "Housing",
                        "NumberOfCredits",
                        "Job",
                        "NumberOfDependents",
                        "Telephone",
                        "ForeignWorker",
                        "CreditRating")
credit_data$CreditRating <- factor(ifelse(credit_data$CreditRating == 1, "good", "bad"))

# feature engineering
credit_data$Gender <- factor(ifelse(credit_data$RelationshipStatusGender == "A92", "female", "male"))

# measuring bias in terms of gender and credit rating
default_rate_female <- credit_data %>%
  filter(Gender == "female") %>%
  select(CreditRating) %>%
  summary()
default_rate_female <- as.numeric(str_extract(default_rate_female[1], "[0-9]+"))/
  (as.numeric(str_extract(default_rate_female[1], "[0-9]+"))+as.numeric(str_extract(default_rate_female[2], "[0-9]+")))

default_rate_male <- credit_data %>%
  filter(Gender == "male") %>%
  select(CreditRating) %>%
  summary()
default_rate_male <- as.numeric(str_extract(default_rate_male[1], "[0-9]+"))/
  (as.numeric(str_extract(default_rate_male[1], "[0-9]+"))+as.numeric(str_extract(default_rate_male[2], "[0-9]+")))

# analysing proxy attributes
chisq.test(credit_data$Gender, credit_data$Purpose)

# building a random forest model
set.seed(100) # setting a random seed to reproduce
training_set_size = 0.7 # what percentage of entire data set used for training
train <- sample(nrow(credit_data), training_set_size*nrow(credit_data), replace = FALSE) # generating the training set
TrainSet <- credit_data[train,]

credit_model <- randomForest(CreditRating ~ ., data = TrainSet, mtry = 4, importance = TRUE)
credit_model

# testing model performance
predictions <- predict(credit_model, credit_data, type = "class") # generate prediction for entire data set
confusion_matrix <- table(predictions, credit_data$CreditRating)
accuracy = mean(predictions == credit_data$CreditRating) # percentage correctly classified

# testing parity and calibration
# parity = is it more likely to falsely judge individuals with
# [some] social identity […] than it is members of some appropriate comparison group.
predictions_actual <- data.frame(credit_data$CreditRating, predictions, credit_data$Gender)
names(predictions_actual) <- c("Actual Rating", "Predicted Rating", "Gender")
View(predictions_actual)
table(predictions_actual$`Predicted Rating`, predictions_actual$`Actual Rating`)

predictions_actual_male <- predictions_actual %>%
  filter(Gender == "male")
predictions_actual_female <- predictions_actual %>%
  filter(Gender == "female")

# false positives = Type I Error: predicted bad risk, actual risk good
# false negatives = Type II Error: predicted good risk, actual risk bad
TP_male <- sum(predictions_actual_male$`Actual Rating` == "bad" & predictions_actual_male$`Predicted Rating` == "bad")
FP_male <- sum(predictions_actual_male$`Actual Rating` == "good" & predictions_actual_male$`Predicted Rating` == "bad")
FN_male <- sum(predictions_actual_male$`Actual Rating` == "bad" & predictions_actual_male$`Predicted Rating` == "good")
TN_male <- sum(predictions_actual_male$`Actual Rating` == "good" & predictions_actual_male$`Predicted Rating` == "good")

TP_female <- sum(predictions_actual_female$`Actual Rating` == "bad" & predictions_actual_female$`Predicted Rating` == "bad")
FP_female <- sum(predictions_actual_female$`Actual Rating` == "good" & predictions_actual_female$`Predicted Rating` == "bad")
FN_female <- sum(predictions_actual_female$`Actual Rating` == "bad" & predictions_actual_female$`Predicted Rating` == "good")
TN_female <- sum(predictions_actual_female$`Actual Rating` == "good" & predictions_actual_female$`Predicted Rating` == "good")

# what percentage of actual bad risk did the model miss?
false_positive_rate <- function (FN, TP) {
  return(FN/(FN+TP))
}

# What percentage of actual good risk did the model miss?
false_negative_rate <- function (FP, TN) {
  return(FP/(FP+TN))
}

ACC_male <- sum(predictions_actual_male$`Actual Rating` == predictions_actual_male$`Predicted Rating`)/nrow(predictions_actual_male)
FPR_male = false_positive_rate(FN_male, TP_male)
FPR_female = false_positive_rate(FN_female, TP_female)

ACC_female <- sum(predictions_actual_female$`Actual Rating` == predictions_actual_female$`Predicted Rating`)/nrow(predictions_actual_female)
FNR_male = false_negative_rate(FP_male, TN_male)
FNR_female = false_negative_rate(FP_female, TN_female)

#
# plot
#

model_stats <- data.frame(
  Accuracy = c(ACC_male, ACC_female),
  FPR = c(FPR_male, FPR_female),
  FNR = c(FNR_male, FNR_female),
  Gender = c("male", "female")
)

acc_plot <- model_stats %>% ggplot(aes(x = Gender, y = Accuracy, fill = Gender)) +
  geom_bar(stat = "identity", width = 0.4) +
  scale_fill_manual(values = c("#2E86C1", "#28B463")) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  xlab("") +
  theme_clean() +
  theme(
    legend.position = "none",
    plot.background = element_rect(color = "white"),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )

fpr_plot <- model_stats %>% ggplot(aes(x = Gender, y = FPR, fill = Gender)) +
  geom_bar(stat = "identity", width = 0.4) +
  scale_fill_manual(values = c("#2E86C1", "#28B463")) +
  ylim(c(0, 0.25)) +
  ylab("False Positive Rate") +
  xlab("") +
  theme_clean() +
  theme(
    legend.position = "none",
    plot.background = element_rect(color = "white"),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )

fnr_plot <- model_stats %>% ggplot(aes(x = Gender, y = FNR, fill = Gender)) +
  geom_bar(stat = "identity", width = 0.4) +
  scale_fill_manual(values = c("#2E86C1", "#28B463")) +
  ylim(c(0, 0.05)) +
  ylab("False Negative Rate") +
  xlab("") +
  theme_clean() +
  theme(
    legend.position = "none",
    plot.background = element_rect(color = "white"),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)
  )

grid.arrange(acc_plot, fpr_plot, fnr_plot, nrow = 1)
```
