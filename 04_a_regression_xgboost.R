# Hent data

# EDA

# Datavask

## recipe
xg_recipe <- recipe(ad_tot_price ~ ., data = finn_train_raw) %>% 
  step_mutate(ad_home_type = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 3)) %>% 
              #bedrooms_missing = is.na(ad_bedrooms)) %>%
  step_rm(ad_id, 
          ad_price,
          ad_tot_price_per_sqm,
          ad_debt, 
          kommune_no, 
          kommune_name, 
          fylke_name, 
          zip_no, 
          zip_name) %>% 
  prep()

## bake
finn_train <- bake(xg_recipe, finn_train_raw)
finn_test  <- bake(xg_recipe, finn_test_raw)



# Tren modell
xg_mod <- boost_tree(mode = "regression") %>% 
  set_engine("xgboost", tree_method = "exact") %>% 
  fit(ad_tot_price ~ ., data = finn_train)



# Test modell
prediction <- predict(xg_mod, finn_test) %>% 
  bind_cols(finn_test_raw) %>% 
  rename(estimate     = .pred, 
         truth        = ad_tot_price) %>% 
  mutate(abs_dev      = abs(truth - estimate),
         abs_dev_perc = abs_dev/truth)


# Evaluer modell

## metrics
prediction %>%
  multi_metric(truth = truth, estimate = estimate) 

# variable importance:
xgboost::xgb.importance(model = xg_mod$fit) %>% 
  xgboost::xgb.ggplot.importance()

# distribution of predicted vs truth
prediction %>% 
  select(estimate, truth) %>% 
  rownames_to_column(var = "id") %>% 
  pivot_longer(-id, names_to = "type", values_to = "value") %>% 
  ggplot(aes(x = value, fill = type)) +
  geom_density(alpha = 0.3)

prediction %>% 
  select(estimate, truth, fylke_name) %>% 
  rownames_to_column(var = "id") %>% 
  pivot_longer(-c(id, fylke_name), names_to = "type", values_to = "value") %>% 
  ggplot(aes(x = value, fill = type)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~fylke_name)

