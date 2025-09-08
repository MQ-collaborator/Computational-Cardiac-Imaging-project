This project was undertaken during a summer studentship with London Medical Schools Medical Research Council (LMS MRC) Computational Cardiac Imaging Group (O'Regan Lab) in summer 2025.
This project built up to a final presentational, discussing clinical factors relevant to the use of image-derived-phenotypes in cariodology and the utility of data science methods employed in a clinical context.

Project title: Classifying Cardiac phenotypes and predicting cardiac function with deep learning

The goal of this project was to use UK biobank data (48 patients, 100 biomarkers each) to train a model to predict systollic blood pressure. For this 2 approaches were taken: conventional  techniques(linear regressions in particular) and deep learning techniques (autoencoder with a linear regression on latent space).
Summary of results: deep learnng provided far superior feature extraction and as a results led us to being able to produce a superior predicton model with clinically interpretable results. 

The most important parts of this project are the model architecture used (regression_autencoder_model.py) a standard MLP. Whilst this internship program is largely educational, the images generated highlight important clinical relationships. 
The relevant training data is subject to data protection (UK biobank) and as such not inclduded. Results and source code can be disseminated .

Usage: run programs from the src directory
