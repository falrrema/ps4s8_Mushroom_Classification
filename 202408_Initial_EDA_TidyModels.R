# Mushroom Classification -------------------------------------------------
# The objectives of this notebooks are:
#
#- Provide an initial exploratory data analysis using R and the best suite of packages for the task. 
#- Based on EDA provide a solution for collapsing categories in categorical features. 
#- Based on EDA provide a solution to missing values.
#- Benchmark models using crossvalidation using tidyverse package. 
#
#Since I am fairly new to tidymodels this challenge is a great way to learn tidymodels potential.
## Set-up ------------------------------------------------------------------
setwd("~/Google Drive/My Drive/DS_Projects/Playground_Series/ps4s8_Mushroom_Classification/")
options(scipen = 9999)

# Loading libraries
library(tidyverse)
library(effectsize)
library(naniar)
library(tidymodels)
library(tune)
library(bonsai)
library(embed)
library(finetune)
library(future)
tidymodels_prefer()

# reading data
train <- read_csv("Data/train.csv")
test <- read_csv("Data/test.csv")
submission <- read_csv("Data/sample_submission.csv")

# Correct column names 
col_names <- colnames(train)
col_names <- gsub("-", "_", col_names)
colnames(train) <- col_names

col_names <- colnames(test)
col_names <- gsub("-", "_", col_names)
colnames(test) <- col_names
features <- train %>% select(-id, -class) %>% colnames()

# To improve visualization of the plots
my_theme <- function() {
  theme_bw() #+
    # theme(
    #   text = element_text(size = 20),
    #   axis.text = element_text(size = 18),
    #   axis.title = element_text(size = 20),
    #   legend.text = element_text(size = 18),
    #   legend.title = element_text(size = 20),
    #   plot.title = element_text(size = 24),
    # )
}

## Missing Data Exploration ------------------------------------------------
# Lets see missing data with naniar package
gg_miss_var(train) + my_theme() + xlab("")

# Missing data Train
# Columns with over 80% missing are: veil_color, stem_root, spore_print_color, veil_type
train_missing <- train %>% 
  map_df(is.na) %>% 
  map_df(sum) %>% 
  pivot_longer(cols = everything(), names_to = "features", values_to = "na_count") %>%
  mutate(pct = round(na_count/nrow(train),3)) %>% 
  arrange(desc(na_count)) 

test_missing  <- test %>% 
  map_df(is.na) %>% 
  map_df(sum) %>% 
  pivot_longer(cols = everything(), names_to = "features", values_to = "na_count") %>%
  gather(features, na_count) %>% 
  mutate(pct = round(na_count/nrow(test),3)) %>% 
  arrange(desc(na_count)) 

# Proportion of missing data are identical between train and test
train_missing %>% 
  left_join(test_missing, by = "features", suffix = c("_train", "_test")) 

# Whats up with veil_type? Almost 95% missing data
train %>% 
  count(veil_type) %>% 
  arrange(desc(n))  %>% 
  left_join(test %>% count(veil_type), by = "veil_type", suffix = c("_train", "_test"))

# Mainly one value, missing could be a category in this case
# This also applies to test

### Does missing have a pattern? --------------------------------------------
# Lets see an upset plot of train
gg_miss_upset(train, text.scale = 2, number.angles = 45, nsets=5)

# **Systematic missingness:**
# 
# The high correlation in missingness among the top five variables suggests a systematic 
# reason for this data being absent. It could be due to data collection methods? 
# certain types of mushrooms being harder to classify? or specific study designs.

### Does missing data have any pattern with the Class? ----------------------
train_bool <- train %>% 
  mutate_at(vars(all_of(features)), function(x) is.na(x))

relevant_feats_missing <- train_bool %>% 
  select(all_of(features)) %>% 
  summarise_all(sum) %>% 
  pivot_longer(cols = everything(), names_to = "feature", values_to = "is_missing") %>%
  filter(is_missing > 5) %>% pull(feature)

results <- train_bool %>%
  select(id, class, all_of(relevant_feats_missing)) %>%
  gather(features, is_missing, all_of(relevant_feats_missing)) %>%
  group_by(features, is_missing) %>%
  # Calculate the proportion of poisonous mushrooms between missing and non missing values
  summarize(total = n(),
            poisonous = sum(class == "p"),
            prop_poisonous = poisonous / total,
            .groups = "drop")  %>% 
  group_by(features) %>%
  # Calculate the difference in proportion of poisonous mushrooms between missing and non missing values
  mutate(baseline = prop_poisonous[!is_missing], 
         abs_diff_from_baseline = abs(prop_poisonous - baseline)) %>% # absolute difference
  filter(is_missing) %>%  # Keep only the rows where is_missing is TRUE
  ungroup() %>%
  arrange(desc(abs(abs_diff_from_baseline)))  # Sort by absolute difference

options(repr.plot.width=16, repr.plot.height=10)
results %>% 
  ggplot(aes(x = reorder(features, abs_diff_from_baseline), y = abs_diff_from_baseline, fill = abs_diff_from_baseline)) +
  geom_bar(stat = "identity", position = "dodge") +
  ylab("Absolute % difference") + xlab("") +
  scale_fill_gradient(low = "#FFCCCC", high = "#990000") +
  ggtitle("Absolute % difference in poisonous mushrooms between missing and non missing values") +
  coord_flip() +
  my_theme() +
  theme(legend.position = "none")

# Missing data is relevant:
#   
#   - `cap_color` has almost 30% difference between missing and non missing values. 
# - `veil_color`and `has_ring` have barely any difference with respect non missing values. 
# 
# This information is relevant when imputing, missing could be category of its own in high missing features. 

## Distribution of numeric Variables ---------------------------------------
num_features <- train %>% 
  select_if(is.numeric) %>% 
  select(-id) %>% colnames()

df_dist <- train %>% 
  select(all_of(num_features)) %>% 
  mutate(type = "train") %>% 
  bind_rows(test %>% # adding test data to compare
              select_if(is.numeric) %>% 
              mutate(type = "test")) %>% 
  pivot_longer(cols = cap_diameter:stem_width, names_to = "features", values_to = "values")

# See distribution
# There is a extreme value in test cap_diameter
df_dist %>% 
  group_by(type, features) %>% 
  summarise(min = min(values, na.rm = TRUE),
            p25 = quantile(values, 0.25, na.rm = TRUE),
            p50 = quantile(values, 0.50, na.rm = TRUE),
            mean = mean(values, na.rm = TRUE),
            p75 = quantile(values, 0.75, na.rm = TRUE),
            max = max(values, na.rm = TRUE)) %>% 
  ungroup() %>% 
  arrange(features)

# Train and test distribution of numeric variables are identical
df_dist %>% 
  group_by(type, features) %>% 
  # Remove extreme ouliers
  filter(values < quantile(values, 0.99, na.rm = TRUE)) %>% 
  ungroup() %>% 
  ggplot(aes(x = type, y = values, fill = type)) +
  geom_boxplot() +
  facet_wrap(~features, ncol=3) +
  my_theme()

# Main ideas:
#   
#   - The distribution between train and test set of the numerical variables are identical. 
# - The distributions are slightly negatively skewed (to the right).
# - The data shows outliers, however `cap_diameter` has a extreme outlier.

## Categorical Variables Exploration ---------------------------------------
# There seems to be a lot of values for each categorical features. I will analyze this by counting each 
# value within a feature, and classifying each value into three types:
#   
#   - Mayority Values that account for the 90% of the Feature.
# - Minority that account for almost the 10% remaining.
# - Rare that account for les than 0.1% 
cat_features <- train %>% 
  select_if(is.character) %>% 
  select(-class) %>% colnames()

cat_counts <- train %>% # If you want to explot test set change dataset here
  select(all_of(cat_features)) %>% 
  pivot_longer(everything(), names_to = "features", values_to = "values") %>%
  group_by(features, values) %>% 
  summarise(n = n()) %>%
  filter(!is.na(values)) %>% # remove NA values from each feature
  arrange(desc(n)) %>% 
  mutate(pct = n/sum(n),
         cum_pct = cumsum(pct),
         type = case_when(
           pct > 0.05 ~ "Majority",
           between(pct, 0.001, 0.05) ~ "Minority",
           TRUE ~ "Rare"
         ))

# Lets see all the categorical variables
cat_counts %>% 
  count(features, type) %>% 
  ggplot(aes(x = type, y = n, fill = type)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = n), vjust = -0.5) +
  facet_wrap(~features, ncol = 3) +
  my_theme() + ylim(c(0,80)) +
  xlab("") + ylab("Counts")

# Example of rare values for cap_surface
cat_counts %>% 
  filter(features == "cap_surface" ) # explore other features 

# Important findings:
#   
#   - `Rare` types make of the mayority of the unique values for almost all of the categorical features.
# - `Rare` types usually have less than 100 values. (Check other features above)
# - `cap_surface`, `habitat` and `cap_shape` have around 70 unique rare values.
# - Many of the rare values in cap_surface is just **garbage or noise**

### Does the original dataset have this rare values? ------------------------
# install.packages("ucimlrepo")
library(ucimlrepo)

# Fetch the Mushroom dataset
mushroom_data <- ucimlrepo::fetch_ucirepo(id = 848)
original_data <- mushroom_data$data$original %>% as_tibble() 

# Correct column names 
col_names <- colnames(original_data)
col_names <- gsub("-", "_", col_names)
colnames(original_data) <- col_names

# Comparing to the original dataset, most of the unique variables seen in train were just created, 
# the original dataset doesn't have this rare values.
original_data %>% count(cap_surface, sort = TRUE)

### Is this true for test set? ------------------------------------------
train_test_count <- train %>% 
  select(-class) %>%
  mutate(split = "train") %>%
  # adding test set to the mix
  bind_rows(test %>% mutate(split = "test"))  %>% 
  select(all_of(cat_features), split) %>% 
  pivot_longer(cols = -split, names_to = "features", values_to = "values")  %>%
  group_by(split, features, values) %>% 
  summarise(n = n()) %>%
  filter(!is.na(values)) %>% # remove NA values from each feature
  arrange(desc(n)) %>% 
  mutate(pct = n/sum(n),
         type = case_when(
           pct > 0.05 ~ "Majority",
           pct >= 0.001 ~ "Minority",
           TRUE ~ "Rare"
         )) 

train_test_count  %>% 
  count(features, type)  %>% 
  pivot_wider(names_from = features, values_from = n)  %>% 
  arrange(type, split)

# Comparing train and test set majority features
train_test_count %>% 
  filter(type == "Minority", # change here to see other types
         features == "habitat") %>% # change here to see other features
  select(split, features, values, pct) %>%
  pivot_wider(names_from = split, values_from = pct) 

# For test set:
#   
# - The majority features are identical between train and test set (explore above).
# - The minority features are identical too.
# - The rar features are different (noise)


### Does these rare values have any predictive power? -----------------------
# Lets calculate chi-square statistic and Cramers'V for feature.
# # Step 1: Create encoding function
encode_categories <- function(data) {
  data %>%
    group_by(features, values) %>%
    summarise(count = n(), .groups = "drop") %>%
    group_by(features) %>%
    mutate(
      total = sum(count),
      prevalence = count / total,
      type = case_when(
        prevalence > 0.05 ~ "Majority",
        prevalence > 0.001 ~ "Minority",
        TRUE ~ "Rare"
      )
    ) %>%
    select(features, values, count, type)
}

# Step 2: Create long dataset in train for categorical variables 
train_long <- train %>% 
  select(class, all_of(cat_features))  %>% 
  pivot_longer(cols = -class, names_to = "features", values_to = "values") %>% 
  filter(!is.na(values)) 

# Step 3: Apply encoding
encoded_categories <- train_long %>%
  encode_categories()

# Step 4: Join encoded categories with original data
encoded_data <- train_long %>%
  left_join(encoded_categories, by = c("features", "values"))

# Identify Features_Types with only 1 categorical variable
# The statistical test to follow can't handle 1 value of a categorical variable
remove_feature_types <- cat_counts %>% 
  count(features, type)  %>% 
  filter(n == 1) %>% 
  unite("features_type", features, type, sep = "_") %>% pull(features_type)

# Step 5: Apply Chi-Square test to identify relationship between categorical variables # nolint
# Cramers V is a measure of association between two categorical variables
# Lowering sample size avoids over-signficance of chi-squared test
set.seed(40)
encoded_data %>%
  unite("features_type", features, type, sep = "_") %>%
  filter(!features_type %in% remove_feature_types) %>% 
  sample_n(5e6) %>% 
  group_by(features_type) %>%
  summarise(
    chi_square = chisq.test(table(values, class))$statistic, # Is there a relationship between the two categorical variables?
    p_value = round(chisq.test(table(values, class))$p.value, 4), # How likely is the relationship due to chance?
    significant = p_value < 0.05,
    cramers_v = cramers_v(table(values, class))$Cramers_v, # How strong is the relationship? 0 to 1
    .groups = "drop") %>%
  mutate(interpretation = case_when(cramers_v < 0.10 ~ "Negligible association",
                                    cramers_v < 0.20 ~ "Weak association",
                                    cramers_v < 0.40 ~ "Moderate association",
                                    cramers_v < 0.60 ~ "Relatively strong association",
                                    cramers_v < 0.80 ~ "Strong association",
                                    cramers_v <= 1.00 ~ "Very strong association",
                                    TRUE ~ "Invalid Cramer's V")) %>%
  arrange(desc(cramers_v))

# Main ideas:
#   
# - Most of the groupings by feature are relevant by chi-squared statistic, however this is expected due to the large sample size. 
# - Cramers'V comes to the rescue which focuses on the effect size, some of the rare groupings are signficant and have strong association.
# - Some rare groupings relevant to notice are `stem_color_Rare`, `stem_surface_Rare`, `veil_color_Rare`.
# - The rest of the rare grouping are not significant or relevant from cramers'V View. 

# Lets check with information value and weight of evidence
iv_data <- encoded_data %>%
  filter(type == "Rare") %>%
  unite("features_type", features, type, sep = "_")  %>% 
  count(features_type, class)  %>% 
  pivot_wider(names_from = class, values_from = n)   %>% 
  mutate(prop_p = p / (p + e), # proportion of poisonous mushrooms
         prop_e = e / (p + e), # proportion of edible mushrooms
         woe = log(prop_e / prop_p), # weight of evidence
         inf_value = (prop_e - prop_p) * woe)  %>% # information value
  arrange(desc(inf_value))  %>% 
  mutate(iv_interpretation = case_when(inf_value < 0.02 ~ "Unpredictive",
                                       inf_value < 0.1 ~ "Weak predictive power",
                                       inf_value < 0.3 ~ "Moderate predictive power",
                                       inf_value < 0.5 ~ "Strong predictive power",
                                       inf_value >= 0.5 ~ "Extremely strong predictive power",
                                       TRUE ~ "Suspicious (potential overfitting)"))

iv_data %>% 
  ggplot(aes(x = reorder(features_type, inf_value), y = inf_value, fill = iv_interpretation)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  xlab("") + ylab("Information Value") +
  ggtitle("Information Value of Rare Categorical Variables") +
  my_theme() 

# Information value on the rare groupings confirm what we did above that `stem_color_Rare`,
# `stem_surface_Rare`, `veil_color_Rare` are relevant to the problem. 
# So in the end its **not recommendable to through away rare values at least for these features.**

### Does test set have the same rare types? ---------------------------------
# Rare types are different between train and test set
train %>% 
  count(stem_color) %>% 
  arrange(desc(n)) %>% 
  full_join(test %>% count(stem_color), by = "stem_color") 

# Collapsing categorical variables ----------------------------------------
# I will collapse rare values to a placeholder called `rare` as I did above using a prevalence of 0.1%,
# this criteria will apply for test set too. 
cat_features <- train %>% 
  select_if(is.character) %>% 
  select(-class) %>% colnames()

# Step 1: Create a function to identify rare categories
identify_rare_categories <- function(data, feature, threshold = 0.001) {
  data %>%
    count(!!sym(feature)) %>%
    mutate(
      total = sum(n),
      prevalence = n / total) %>%
    filter(prevalence >= threshold) %>%
    pull(!!sym(feature))
}

# Step 2: Create a function to replace rare categories with "rare"
replace_rare_categories <- function(data, feature, threshold = 0.001) {
  non_rare <- identify_rare_categories(data, feature, threshold)
  data %>%
    mutate(!!sym(feature) := if_else(!!sym(feature) %in% non_rare, 
                                     !!sym(feature), 
                                     "rare"))
}

# Step 3: Apply the transformation to all categorical variables in the training set
# The reduce function in Step 3 Applies the replace_rare_categories function to each categorical feature in turn
# The result is a dataset where all rare categories (prevalence < 0.001) in all categorical variables have been replaced with "Rare"
train_collapsed <- cat_features %>%
  reduce(function(data, feature) replace_rare_categories(data, feature), .init = train)

# Step 4: Apply the transformation to all categorical variables in the test set
test_collapsed <- cat_features %>%
  reduce(function(data, feature) replace_rare_categories(data, feature), .init = test)

# Lets check if the rare values are now "rare"
train_collapsed %>% 
  count(stem_color) %>% 
  arrange(desc(n)) %>% 
  full_join(test_collapsed %>% count(stem_color), by = "stem_color")

# Imputing missing values -------------------------------------------------
# In this first submission I will just create a `Missing` category for categorical variables and for
# numerical variables impute the median. For logical columns will convert them to integer. 
impute_missing <- function(data) {
  # Identify numeric and categorical columns
  num_cols <- data %>% select_if(is.numeric) %>% names()
  cat_cols <- data %>% select_if(is.character) %>% names()
  
  # Calculate median for numeric columns
  medians <- data %>% summarise(across(all_of(num_cols), ~median(.x, na.rm = TRUE)))
  
  # Impute missing values
  data %>%
    mutate(across(all_of(num_cols), ~replace_na(.x, medians[[cur_column()]]))) %>%
    mutate(across(all_of(cat_cols), ~replace_na(.x, "Missing")))
}

# Apply to train data
train_imputed <- train_collapsed %>% 
  mutate_if(is.logical, as.integer) %>%
  impute_missing()

# Apply to test data
test_imputed <- test_collapsed %>% 
  mutate_if(is.logical, as.integer) %>%
  impute_missing()

# Example of veil type for train and test 
train_imputed %>% 
  count(veil_type) %>% 
  left_join(test_imputed %>% count(veil_type), by = "veil_type", suffix = c("_train", "_test"))

# Modeling with TidyModels ------------------------------------------------
# Lets start by preparing the target as passing characters as factors
train_ready <- train_imputed %>% 
  mutate(class = as.factor(as.integer(class == 'p'))) %>%
  mutate_if(is.character, as.factor) 

test_ready <- test_imputed %>% 
  mutate_if(is.character, as.factor)

write_csv(train_ready, "Data/train_ready_1s.csv")
write_csv(test_ready, "Data/test_ready_1s.csv")

# 1. Split the data
data_split <- initial_split(train_ready, prop = 0.7)
train_data <- training(data_split)
test_data <- testing(data_split)

# 2. Define two different recipes
recipe_no_encoding <- recipe(class ~ ., data = train_data) %>%
  step_rm(id) %>%
  step_YeoJohnson(all_numeric_predictors())%>%
  step_novel(all_nominal_predictors())%>%
  step_zv()

recipe_effect_encoding <- recipe(class ~ ., data = train_data) %>%
  step_rm(id) %>%
  step_YeoJohnson(all_numeric_predictors())%>%
  step_novel(all_nominal_predictors())%>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(class)) %>% 
  step_zv()

# 3. Define the LightGBM model with tunable parameters
lightgbm_model_tune <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm",
             is_unbalance = TRUE,
             early_stop = 50) %>%
  set_mode("classification")

# 4. Create a workflow set
workflow_exp <- workflow_set(
  preproc = list(no_encoding = recipe_no_encoding, effect_encoding = recipe_effect_encoding),
  models = list(lightgbm = lightgbm_model_tune),
  cross = FALSE
)

# 5. Define the resampling method
cv_folds <- vfold_cv(train_data, v = 5, strata = class)

# 6. Define metrics
metrics <- metric_set(mcc)

# 7. Define the parameter space
lgb_params <- parameters(
  trees(range = c(300, 2000)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-4, -1), trans = log10_trans()),
  min_n(range = c(30, 1000)),
  sample_size = sample_prop(range = c(0.4, 1)),
  mtry(range = c(5,21))
)

# 8. Set up Bayesian optimization
bo_ctrl <- control_bayes(
  no_improve = 20,
  time_limit = 3600,
  verbose = TRUE,
  verbose_iter = TRUE
)

# 9. Set parallel processing
plan(multisession, workers = 8)

# 10. Tune the workflows using Bayesian optimization
tuned_workflows <- workflow_exp %>%
  workflow_map(
    fn = "tune_bayes",
    resamples = cv_folds,
    iter = 100,  # Increased due to more parameters
    initial = 10,  # Increased due to more parameters
    metrics = metrics,
    control = bo_ctrl,
    param_info = lgb_params
  )
