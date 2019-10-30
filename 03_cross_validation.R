
# An example of how to set up cross validation

library(tidyverse)
library(parsnip)
library(recipes)
library(rsample)
library(readxl)
library(writexl)
library(xgboost)

finn <- ads %>% 
  select(
    ad_id,
    ad_owner_type,
    ad_home_type,
    ad_bedrooms,
    ad_sqm,
    ad_expense,
    ad_tot_price,
    lat,
    lng,
    kommune_name,
    avg_income,
    avg_fortune,
    ad_built,
    ad_floor
  ) %>%   
  mutate(ad_home_type  = fct_lump(ad_home_type, 4),
         ad_owner_type = fct_lump(ad_owner_type, 2),
         kommune_name  = fct_lump(kommune_name, 220))

finn_recipe <- finn %>% 
  recipe(ad_tot_price ~ .) %>% 
  step_rm(ad_id) %>% 
  step_log(ad_sqm, ad_tot_price) %>% 
  step_meanimpute(all_numeric()) %>% 
  step_modeimpute(all_nominal()) %>% 
  step_integer(ad_home_type, ad_owner_type, kommune_name)

set.seed(42)

# Create 10 folds of data
ad_split <- vfold_cv(finn, 10) %>% 
  mutate(recipe      = map(splits, prepper, recipe = finn_recipe, retain = FALSE),
         train_raw   = map(splits, training),
         test_raw    = map(splits, testing),
         ad_id       = map(test_raw, ~select(.x, ad_id)),
         train       = map2(recipe, train_raw, bake),
         test        = map2(recipe, test_raw, bake)) %>% 
  select(-test_raw)

#Setup models
mod_linear_reg <- function(x) {
  linear_reg() %>% 
    set_engine("lm") %>% 
    fit(ad_tot_price ~ ., x)  
}


mod_rand_forest <- function(x) {
  rand_forest(trees = 100, mode = "regression") %>%
    set_engine("ranger") %>% 
    fit(ad_tot_price ~ ., x) 
}

mod_xgb <- function(x) {
  boost_tree(trees = 250, 
             mode = "regression", 
             learn_rate = 0.3, 
             tree_depth = 7) %>%
    set_engine("xgboost") %>% 
    fit(ad_tot_price ~ ., x) 
}

# Train models on all folds
ad_mod <- ad_split %>% 
  mutate(mod_linear_reg  = map(train, mod_linear_reg),
         mod_rand_forest = map(train, mod_rand_forest),
         mod_xgb         = map(train, mod_xgb))

# Get predictions 
ad_pred <- ad_mod %>% 
  mutate(pred_linear_reg  = map2(mod_linear_reg, test, predict),
         pred_rand_forest = map2(mod_rand_forest, test, predict),
         pred_xgb         = map2(mod_xgb, test, predict))

#Calculate predictions for each fold
ad_pred_fold <- ad_pred %>%
  select(ad_id, train, test, pred_linear_reg, pred_rand_forest, pred_xgb) %>%
  mutate(
    mape_linaer_reg  = map2_dbl(test, pred_linear_reg, 
                                ~ mape_vec(exp(.x$ad_tot_price),
                                           exp(.y$.pred))),
    mape_rand_forest = map2_dbl(test, pred_rand_forest, 
                                ~ mape_vec(exp(.x$ad_tot_price), 
                                           exp(.y$.pred))),
    mape_rand_xgb    = map2_dbl(test, pred_xgb, 
                                ~ mape_vec(exp(.x$ad_tot_price), 
                                           exp(.y$.pred)))
  )

# Get mean prediciton for each fold
ad_pred_fold %>% 
  summarise_at(vars(matches("mape")), mean)

# For hyperparameter tuning - send in different variatons of the same model

