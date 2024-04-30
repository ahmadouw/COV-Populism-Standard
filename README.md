# PopAut: An Annotated Corpus for Populism Detection in Austrian News Comments

This repository highlights the sampling process for the annotated sample published in "PopAut: An Annotated Corpus for Populism Detection in Austrian News Comments", as well as the experiments conducted. The data set is available under the CC BY-NC-ND 4.0 license upon request at https://doi.org/10.48436/vbkwj-b8t85.

**Authors:** Ahmadou Wagne, Julia Neidhardt, Thomas Elmar Kolb

**Abstract:** Populism is a phenomenon that is noticeably present in the political landscape of various countries over the past decades. While populism expressed by politicians has been thoroughly examined in the literature, populism expressed by citizens is still underresearched, especially when it comes to its automated detection in text. This work presents the PopAut corpus, which is the first annotated corpus of news comments for populism in the German language. It features 1,200 comments collected between 2019-2021 that are annotated for populist motives anti-elitism, people-centrism and people-sovereignty. Following the definition of Cas Mudde, populism is seen as a thin ideology. This work shows that annotators reach a high agreement when labeling news comments for these motives. The data set is collected to serve as the basis for automated populism detection using machine-learning methods. By using transformer-based models, we can outperform existing dictionaries tailored for automated populism detection in German social media content. Therefore, our work provides a rich resource for future work on the classification of populist user comments in the German language.

## Content:
### Code Directory
The directory `code` contains all necessary code to comprehend the conducted sampling, analysis, and experiments. The raw data to reproduce the experiment is not publicly available. To create a virtual environment (using conda) with the necessary requirements simply use:
```bash
conda env create -f requirements.yml
```

## Notebooks

### data_exploration.ipynb
- Descriptive analysis of the full data from January 2019 to November 2021
- `helper_functions.py` contains code for reading the data and calculating descriptive statistics
- **Requires:** Full sample of all comments in the observed time frame, separate file containing the list of keywords for every article

### annotation_sample.ipynb
- Describes the sampling process for the final annotation sample
- The final sample contains 400 comments from each sub-sample (COVID-19, non-COVID-19, reference)
- **Requires:** COVID-19, non-COVID-19, reference samples

### label_processing.ipynb
- Processes the output of the lime survey annotation study
- Furthermore creates the final published data set
- `lime_processing.py` contains code to transform the labels, calculate the agreement and create the final annotated corpus

### baselines.ipynb
- Sampling process for the training and test samples for all experiments
- Implementation of the Gründl and the R&P dictionary
- Evaluation on the test set with different dictionary scores (populist label assigned with the occurrence of one or two dictionary terms)
- `baseline_models.py` contains code for the model training and evaluation
- `data_preparation.py` specifies sampling and pre-processing for the baselines
- **Requires:** COVID-19, non-COVID-19, reference samples, annotated sample

### baseline_extension.ipynb
- Implementation and evaluation of logistic regression, linear SVC, random forest
- **Requires:** Training sample, test sample
