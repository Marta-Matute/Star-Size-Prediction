# Pràctica Kaggle APC UAB 2021-2022
### Marta Matute
### DATASET: [Star Dataset: Stellar Classification](https://www.kaggle.com/vinesmsuic/star-categorization-giants-and-dwarfs?select=Star39552_balanced.csv)

## Summary
The dataset with which we'll be working uses data extracted from [Vizier](https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I%2F239%2Fhip_main&-out.max=50&-out.form=HTML+Table&-out.add=_RAJ%2C_DEJ&-sort=_r&-oc.form=sexa) catalogue access tool, as well as the The Hipparcos and Tycho Catalogues.

In the Kaggle webpage we find 4 different csv files. Two of them are already preprocessed data. Since one of the goals for this project is to learn how to properly preprocess a dataset, we will be working with the non-preprocessed data. Thus we will be using the file *Star99999_raw.csv*. 

The former contains a total of 99999 instances and 6 attributes. All of them are numerical. 

### Dataset Goals
We will attempt to create a model to correctly classify a star as either being a *giant* star or a *dwraf* star. This dichotomous classification is a broad generalization for stellar classification. Stars are usually classified in 7 different categories, where the first 3 are types of giant stars, and the other 4 types of dwarfs. Our model will be a simpler version of this classification, separating only between dwarfs and giants. 

### Model results
| Model                    | Acc before hypertuning |  Acc after Hypertuning | Test Results |
|--------------------------|----------|------------ |-----------|
| Logistic Regression      | 0.951391 | 0.951641    |  0.919875 | 
| K-Neighbors Classifier   | 0.941496 |    --       | --|
| Decision Tree Classifier | 0.865001 |     --      | --|
| XGBoost                  | 0.959365 | 0.961630    | 0.922236 |
| Random Forest            | 0.956828 | 0.958965    | 0.921490|
| Ensemble                 |    --    | 0.960721    | 0.921988|

## Conclusions and future work
This dataset gave really good results from the beggining, even before we tuned or models or used the ensemble model. Even then, it was still a really workable and flexible dataset that offered room for new variables to work with.

I order to improve this classification, there are a couple things we could do. In terms of approach, we could try to create a model that would not only classify between dwarf and giants, but also specify further and determine the type of dwarf or the type of giant that the star is actually classified as. 

In terms of modeling, we could have tried our models with a different preprocessed data (for example with min-max normalization instead of 0 to 1). It would also be a good idea to find the best parameters for the `N-Neighbours Clasifier` and add it to the voting ensemble, although the results would probably be the same or insignicantly different. Lastly, we could have tried with a different type of ensemble; we tried with a *soft* voting system, meaning  that the final decision was made using the average score for all models. We could have tried a *hard* voting system to potencially get different results.

As well, we could have tried more models, although that improvement offers a list of possibilities that would never end. 

In the end, this dataset was easy to work with and has been really helpful to learn more about classification models.
