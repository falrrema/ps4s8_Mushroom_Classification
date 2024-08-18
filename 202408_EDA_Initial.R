###############################################
# Mushroom Classification - Playground Series #
###############################################

setwd("~/Documents/PS_s4e8")
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
table_na(train) # Has a lot of missing value
table_na(test) # Same

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
  # Remove extreme ourliers
  filter(values < quantile(values, 0.99, na.rm = TRUE)) %>% 
  ungroup() %>% 
  ggplot(aes(x = type, y = values, fill = type)) +
  geom_boxplot() +
  facet_wrap(~features, ncol=3)

# Categorical Variables Exploration ---------------------------------------
cat_features <- train %>% 
  select_if(is.character) %>% 
  select(-class) %>% colnames()

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

cat_counts <- train %>% 
  select(all_of(cat_features)) %>% 
  gather(features, values, everything()) %>% 
  count(features, values)

# Creating function to convert levels 
convert_unk <- function(x, cut_n = 10) {
  to_unk <- names(table(x)[table(x) <= cut_n])
  to_unk <- paste0(to_unk, collapse = "|")
  x <- gsub(to_unk, "unk", x)
  x
}



