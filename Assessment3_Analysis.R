# Loading libraries

library(tidyverse)
library(tidymodels)
library(kableExtra)
library(skimr)
library(gmodels)
library(plotly)
library(patchwork)
library(vip)
library(inspectdf)
library(vcd)
library(ggmosaic)
library(grkmisc)

# Loading the data

survey = read.csv("Survey Data/survey_results_public.csv")
head(survey)

# Exploratory Data Analysis

skim_without_charts(survey)


survey_subset = survey %>%
  select("ResponseId","Employment", "LearnCode", "YearsCode", "Gender",
         "WorkExp", "EdLevel","Country", "Age", "Ethnicity")%>%
  na.omit()

## Data Cleaning

survey_subset = survey_subset %>%
  mutate(Employment = case_when(grepl("^Employed", Employment) ~ "Employed",
                                grepl("Not employed", Employment) ~ "Not Employed",
                                TRUE ~ "Other")) %>%
  mutate(Gender = case_when(grepl("Non-binary", Gender) ~ "Other",
                            grepl("^Man", Gender) ~ "Man",
                            grepl("^Woman", Gender) ~ "Woman",
                            TRUE ~ "Other")) %>%
  mutate(LearnCode = case_when(grepl("University|university", LearnCode) ~ "University",
                               grepl("Bootcamp|bootcamp", LearnCode) ~ "Bootcamp",
                               grepl("Online|online", LearnCode) ~ "Online Course",
                               TRUE ~ "Other")) %>%
  mutate(EdLevel = case_when(grepl("^Bachelor", EdLevel) ~ "Bachelor degree",
                             grepl("^Master", EdLevel) ~ "Master's degree",
                             grepl("doctor", EdLevel) ~ "Doctoral degree",
                             grepl("Associate", EdLevel) ~ "Associate degree",
                             grepl("Secondary", EdLevel) ~ "Secondary school",
                             grepl("Primary", EdLevel) ~ "Primary school",
                             grepl("Professional", EdLevel) ~ "Professional degree",
                             TRUE ~ "Other")) %>%
  mutate(Ethnicity = case_when(grepl("own words", Ethnicity) ~ "Other",
                               str_count(Ethnicity, "\\w+") > 1 ~ "Multiracial",
                               TRUE ~ Ethnicity)) %>%
  mutate(YearsCode = as.factor(YearsCode)) %>%
  mutate(Age = as.factor(Age))%>%
  mutate(Employment = as.factor(Employment))



## Categorical variables
survey_subset %>%
  inspect_cat()%>%
  show_plot(col_palette = 1)

## Numerical variables

survey_subset %>%
  select("WorkExp") %>%
  inspect_num()%>%
  show_plot(col_palette = 1)



## Multivariate analysis

### Employment vs age

cbPalette <- c("#EBB233", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#E69F00")
survey_subset %>% 
  ggplot(aes(x = Employment, fill = Age)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = cbPalette) +
  labs( y = "Proportion") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

### Employment vs ed level
survey_subset %>% 
  ggplot(aes(x = Employment, fill = EdLevel)) +
  geom_bar(position = "fill")+
  scale_fill_manual(values = cbPalette, name = "Education Level") +
  labs(y = "Proportion")+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

### Employment vs code
survey_subset %>% 
  drop_na("LearnCode")%>%
  ggplot(aes(x = Employment, fill = LearnCode)) +
  geom_bar(position = "fill")+
  scale_fill_manual(values = cbPalette, name = "Learn Code") +
  labs(y = "Proportion")+
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

### Employment vs work experience
ggplot(data = survey_subset, aes(x = Employment, y = WorkExp, fill = Employment)) +
  geom_boxplot() +
  xlab("Employment") + 
  ylab("Work Experience") +
  #ylim(0,2)+ # Limit y scale to see the boxes for each year
  scale_fill_manual(values = cbPalette)+
  #ggtitle("Sales in North America by Year") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
  )


### Employment vs gender

ggplot(data = survey_subset) +
  geom_mosaic(aes(x = product(Gender, Employment), fill = Gender), offset = 0.02) +
  scale_fill_manual(values = cbPalette) +
  labs(y="Gender", x="Employment", title = "Employment per gender") +
  theme(axis.text.x = element_text(angle = 90),
        legend.position = "right",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        panel.background = element_blank())

## Model

### Preprocessing

set.seed(4321)
survey_split <- initial_split(survey_subset)
survey_split

survey_train <- training(survey_split)
survey_test <- testing(survey_split)


survey_recipe <- recipe(Employment ~ ., data = survey_train) %>%
  step_rm(ResponseId) %>%
  step_other(Country, YearsCode) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_predictors()) %>%
  step_zv(all_predictors()) %>% #Remove predictors with variance = 0
  step_corr(all_predictors()) %>%
  prep()

survey_train_preproc <- bake(survey_recipe, new_data = NULL)

survey_test_preproc <- bake(survey_recipe, new_data = survey_test)

### Logistic regression

survey_test_preproc <- bake(survey_recipe, new_data = survey_test)

#### Cross-validation data
survey_cv <- vfold_cv(survey_train_preproc, v = 10)

logistic_cv <- fit_resamples(logistic_spec, Employment ~., survey_cv)

logistic_cv %>%
  collect_metrics()
### Random forest

rf_spec <- rand_forest( 
  mode = "classification"
) %>% 
  set_engine( "ranger")

set.seed(4321)
rf_cv <- fit_resamples(rf_spec, Employment ~ . , survey_cv)
rf_cv %>%
  collect_metrics()

## Model evaluation

survey_logistic <- logistic_spec %>%
  fit(Employment ~., data = survey_train_preproc)
survey_logistic

### Variable of importance plot
survey_logistic %>%
  vip(all_permutations = TRUE, mapping = aes_string(fill = "Importance")) +
  scale_fill_distiller(palette = "Reds", direction = 1)+
  ggtitle("Variable Importance Scores for predictors") + 
  theme_bw()

### Prediction
survey_preds <- predict(survey_logistic,
                        new_data = survey_test_preproc) %>%
  bind_cols(survey_test_preproc %>% select(Employment))

survey_preds%>%
  head()

survey_preds %>%
  metrics(truth = Employment, estimate = .pred_class)
  
