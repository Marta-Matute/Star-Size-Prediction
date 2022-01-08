# Pràctica Kaggle APC UAB 2021-2022
### Marta Matute
### DATASET: [Star Dataset: Stellar Classification](https://www.kaggle.com/vinesmsuic/star-categorization-giants-and-dwarfs?select=Star39552_balanced.csv)

## Resum
The dataset with which we'll be working uses data extracted from [Vizier](https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=I%2F239%2Fhip_main&-out.max=50&-out.form=HTML+Table&-out.add=_RAJ%2C_DEJ&-sort=_r&-oc.form=sexa) catalogue access tool, as well as the The Hipparcos and Tycho Catalogues.

In the Kaggle webpage we find 4 different csv files. Two of them are already preprocessed data. Since one of the goals for this project is to learn how to properly preprocess a dataset, we will be working with the non-preprocessed data. Thus we will be using the file *Star99999_raw.csv*. 

The former contains a total of 99999 instances and 6 attributes. All of them are numerical. 

### Dataset Goals
We will attempt to create a model to correctly classify a star as either being a *giant* star or a *dwraf* star. This dichotomous classification is a broad generalization for stellar classification. Stars are usually classified in 7 different categories, where the first 3 are types of giant stars, and the other 4 types of dwarfs. Our model will be a simpler version of this classification, separating only between dwarfs and giants. 

### Model results before hyperparameter tuning

| Model                    |     Mean |        Std |
|--------------------------|----------|------------|
| Logistic Regression      | 0.951391 | 0.00393402 |
| K-Neighbors Classifier   | 0.941496 | 0.00421952 |
| Decision Tree Classifier | 0.864777 | 0.00601723 |
| XGBoost                  | 0.959365 | 0.00312391 |
| Random Forest            | 0.956486 | 0.00367121 |

## Conclusions

## Potential improvements
It would be of great use to create a model to classify a star within the 7 different size categories we mentioned earlier. Although classifying as either dwarf or giant is already good, fine tuning this result by giving the type of dwarf or the type of giant would be a good improvement.
