# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data.csv')

# Handling missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Encoding categorical data
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased, 
                         levels = c('No', 'Yes'),
                         labels = c(0, 1))
                          
# splitting the data into training set and test set

# install.packages('caTools') # installs caTools library
# library(caTools) # select the library caTools

set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
#factors in R cannot be scaled, hence we use only 2 (age) and 3 (salary) columns to scale
training_set[, 2 : 3] = scale(training_set[, 2 : 3])
test_set[, 2 : 3] = scale(test_set[, 2 : 3])