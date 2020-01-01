# About ETD Autoclassifier
A Flask API for classifying Electronic Theses and Dissertations prepared by students at [The University of Zambia](http://www.unza.zm).

## Classification Models

Empirical evaluation was conducted to determine optimal features and, additionally, effective estimators.

###  ETD type classifier
* Estimators
    * RandomForests
* Feature extraction 
    * (i) Text on coverpages 
    * (ii) Number of pages in manuscript
    * (iii) Text on coverpages + Number of pages
* Feature selection
    * Text on cover pages (Countvectorizer)

### ETD collection classifier
* Estimators 
    * SDG
* Feature extraction 
    * (i) Manuscript title 
    * (ii) Manuscript abstract
    * (iii) Manuscript title + abstract
* Feature selection
    * Manuscript title + abstract
