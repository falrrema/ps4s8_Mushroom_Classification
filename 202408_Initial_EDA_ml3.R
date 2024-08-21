###############################################
# Mushroom Classification - Playground Series #
###############################################

setwd("~/Google Drive/My Drive/DS_Projects/Playground_Series/ps4s8_Mushroom_Classification/")
# Set-Up ------------------------------------------------------------------
library(tidyverse)

train = read_csv("Data/train.csv")
test = read_csv("Data/test.csv")

# Correct column names 
col_names <- colnames(train)
col_names <- gsub("-", "_", col_names)
colnames(train) <- col_names

col_names <- colnames(test)
col_names <- gsub("-", "_", col_names)
colnames(test) <- col_names
features <- train %>% select(-id, -class) %>% colnames()

# Missing Data Exploration ------------------------------------------------
glimpse(train)

# Missing data 
train %>% 
  map_df(is.na) %>% 
  map_df(sum) %>% 
  gather(features, na_count) %>% 
  mutate(pct = na_count/nrow(train)) %>% 
  arrange(desc(na_count)) %>% 
  print(n = 22)

test %>% 
  map_df(is.na) %>% 
  map_df(sum) %>% 
  gather(features, na_count) %>% 
  mutate(pct = na_count/nrow(test)) %>% 
  arrange(desc(na_count)) %>% 
  print(n = 22)

# Whats up with veil_type
train %>% count(veil_type) %>% 
  arrange(desc(n)) %>% print(n = 23)

test %>% count(veil_type) %>% 
  arrange(desc(n)) %>% print(n = 23)

# Does missing data have any pattern with the Class?
train_bool <- train %>% 
  mutate_at(vars(all_of(features)), function(x) !is.na(x))

relevant_feats_missing <- train_bool %>% 
  select(all_of(features)) %>% 
  summarise_all(sum) %>% 
  gather() %>% 
  mutate(is_missing = nrow(train_bool) - value) %>% 
  filter(is_missing > 5) %>% pull(key)

results <- train_bool %>% 
  select(id, class, all_of(relevant_feats_missing)) %>% 
  gather(features, not_missing, all_of(relevant_feats_missing)) %>% 
  count(features, not_missing, class) %>% 
  spread(class, n) %>% 
  mutate(pct_bad = p/(p+e)) # yes it does

results %>% 
  ggplot(aes(x = features, y = pct_bad, fill = not_missing)) +
  geom_bar(stat = "identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1") +
  theme(axis.text.x = element_text(angle = 45, vjust = 0.8)) 

# Distribution of numeric Variables ---------------------------------------
num_features <- train %>% 
  select_if(is.numeric) %>% 
  select(-id) %>% colnames()

df_dist <- train %>% 
  select(all_of(num_features)) %>% 
  mutate(type = "train") %>% 
  bind_rows(test %>% 
              select_if(is.numeric) %>% 
              mutate(type = "test")) %>% 
  gather(features, values, cap_diameter:stem_width)

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
  facet_wrap(~features, ncol=3)

# Categorical Variables Exploration ---------------------------------------
cat_features <- train %>% 
  select_if(is.character) %>% 
  select(-class) %>% colnames()

cat_counts <- train %>% 
  select(all_of(cat_features)) %>% 
  gather(features, values, everything()) %>% 
  count(features, values)

# Lets analyze just cap_color
cat_counts %>% 
  filter(features == "cap_color") %>% 
  arrange(desc(n)) %>% 
  print(n = 70)

train %>% 
  select(class, cap_color) %>% 
  count(cap_color, class) %>% 
  group_by(cap_color) %>% 
  arrange(cap_color, desc(n)) %>% 
  filter(n > 1) %>% 
  spread(class, n) %>% 
  mutate(pct_poison = p / (p+e)) %>% 
  arrange(desc(e+p)) %>% 
  print(n = 70)

# Lets analyze stem_surface
cat_counts %>% 
  filter(features == "stem_surface") %>% 
  arrange(desc(n)) %>% 
  print(n = 70)

test %>% 
  count(stem_surface) %>% 
  arrange(desc(n)) %>% 
  print(n = 70)

train %>% 
  select(class, stem_surface) %>% 
  count(stem_surface, class) %>% 
  group_by(stem_surface) %>% 
  arrange(stem_surface, desc(n)) %>% 
  filter(n > 1) %>% 
  spread(class, n) %>% 
  mutate(pct_poison = p / (p+e)) %>% 
  arrange(desc(e+p)) %>% 
  print(n = 70)

# Lets analyze cap_surface
cat_counts %>% 
  filter(features == "cap_surface") %>% 
  arrange(desc(n)) %>% 
  print(n = 70)

test %>% 
  count(cap_surface) %>% 
  arrange(desc(n)) %>% 
  print(n = 70)

train %>% 
  select(class, cap_surface) %>% 
  count(cap_surface, class) %>% 
  group_by(cap_surface) %>% 
  arrange(cap_surface, desc(n)) %>% 
  filter(n > 1) %>% 
  spread(class, n) %>% 
  mutate(pct_poison = p / (p+e)) %>% 
  arrange(desc(e+p)) %>% 
  print(n = 70)

# Collapsing categorical variables ----------------------------------------
# Function to collapse rare categories for a single column
collapse_rare_categories_col <- function(col, threshold) {
  freq <- table(col)
  rare_levels <- names(freq[freq < threshold])
  ifelse(col %in% rare_levels, "unk", as.character(col))
}

cat_features <- train %>% 
  select_if(is.character) %>% 
  select(-class) %>% colnames()

# Apply the collapsing function to each categorical column
train[cat_features] <- map(train[cat_features], ~collapse_rare_categories_col(.x, 20))
test[cat_features] <- map(test[cat_features], ~collapse_rare_categories_col(.x, 20))


train %>% 
  select(class, has_ring) %>% 
  count(has_ring, class) %>% 
  group_by(has_ring) %>% 
  arrange(has_ring, desc(n)) %>% 
  filter(n > 1) %>% 
  spread(class, n) %>% 
  mutate(pct_poison = p / (p+e)) %>% 
  arrange(desc(e+p)) %>% 
  print(n = 70)

test %>% 
  count(stem_surface) %>% 
  arrange(desc(n)) %>% 
  print(n = 70)

# Imputing missing values ------------------------------------------------
# Will impute the missing values of each feature with the category that is most similar 
# by class proportions

create_imputation_map <- function(data, feature_cols, class_col) {
  imputation_map <- map(feature_cols, function(feature_col) {
    proportions <- data %>%
      group_by(!!sym(feature_col)) %>%
      summarise(total = n(),
                positive = sum(!!sym(class_col) == "p"),
                proportion = positive / total) %>%
      arrange(proportion)
    
    na_prop <- proportions$proportion[is.na(proportions[[feature_col]])]
    if (length(na_prop) > 0) {
      closest_category <- proportions %>%
        filter(!is.na(!!sym(feature_col))) %>%
        mutate(diff = abs(proportion - na_prop)) %>%
        arrange(diff) %>%
        slice(1) %>%
        pull(!!sym(feature_col))
      
      return(closest_category)
    } else {
      return(NULL)
    }
  })
  
  set_names(imputation_map, feature_cols)
}


apply_imputation_map <- function(data, imputation_map) {
  for (col in names(imputation_map)) {
    if (!is.null(imputation_map[[col]])) {
      data[[col]] <- ifelse(is.na(data[[col]]), imputation_map[[col]], data[[col]])
    }
  }
  data
}

calculate_median_map <- function(data) {
  data %>%
    select(where(is.numeric)) %>%
    summarise(across(everything(), median, na.rm = TRUE)) %>%
    as.list()
}

impute_median <- function(data, median_map) {
  data %>%
    mutate(across(names(median_map), 
                  ~ifelse(is.na(.), median_map[[cur_column()]], .)))
}

# Imputing categorical features
features_to_impute <- c(cat_features, "does_bruise_or_bleed",  "has_ring")
imputation_map <- create_imputation_map(train, features_to_impute, "class")
train <- apply_imputation_map(train, imputation_map)
test <- apply_imputation_map(test, imputation_map)

# Imputing numeric features
median_map <- calculate_median_map(train %>% select(-id))
train <- impute_median(train, median_map)
test <- impute_median(test, median_map)
glimpse(train)
glimpse(test)

write_csv(train, "Data/train_clean_inputed.csv")
write_csv(test, "Data/test_clean_inputed.csv")

# Modeling ----------------------------------------------------------------
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3verse)
library(mlr3extralearners)
library(mlr3viz)

# Convert class column to binary and character to factor and logical to binary
train_formatted <- train %>% 
  mutate(class = factor(as.integer(class == 'p'))) %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.logical, as.integer)

test_formatted <- test %>% 
  mutate_if(is.character, as.factor) %>% 
  mutate_if(is.logical, as.integer)

# Create the task
task <- TaskClassif$new(id = "mushroom", backend = train_formatted, target = "class")

# Create the preprocessing pipeline
impact_encode <- po("encodeimpact")

# Add learners
learners <- list(
lrn("classif.xgboost", eval_metric = "logloss"),
#lrn("classif.lightgbm", metric = "binary_logloss"),
lrn("classif.cv_glmnet", alpha = 1),  # Lasso
lrn("classif.cv_glmnet", alpha = 0.5)  # Elastic Net
)

# Create learners with impact encoding
learners_encoded <- map(learners, function(lrn) {
  pipeline <- impact_encode %>>% po(lrn)
  GraphLearner$new(pipeline)
})

# Combine all learners
all_learners <- c(learners_encoded)

# Set unique IDs for all learners
learner_names <- c("xgboost", "lasso", "elasticnet")
for (i in seq_along(all_learners)) {
  if (i <= length(learners)) {
    all_learners[[i]]$id <- learner_names[i]
  } else {
    all_learners[[i]]$id <- paste0(learner_names[i - length(learners)], "_encoded")
  }
}

# Set up the benchmark design
design <- benchmark_grid(
  tasks = task,
  learners = all_learners,
  resamplings = rsmp("cv", folds = 5)
)

# Run the benchmark
bmr <- benchmark(design)
