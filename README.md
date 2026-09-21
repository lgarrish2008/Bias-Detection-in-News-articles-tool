# BIAS DETECTION IN ARTICLES NLP PROJECT

An NLP-based Python application exploring whether political bias in individual news articles can be computationally quantified.

Artificial Ignorance takes the URL of a news article, extracts and preprocesses its text, converts its language into numerical TF-IDF features, and applies a trained regression model to produce a political-bias score on a 0–100 scale.

0 = Left  |  50 = Centre  |  100 = Right

Built as my OCR A-Level Computer Science programming project.

## Why I built it

Most media-bias services classify entire publications rather than individual articles. I wanted to explore whether the language of an article itself could instead be analysed computationally.

The project also became an experiment in a broader question: **how far can a qualitative and politically contested concept such as bias be reduced to quantitative data?**

Rather than treating the model's output as objective truth, the application is designed to make both its prediction and its limitations visible to the user.

## How it works

Article URL
->
Web scraping
->
Text cleaning and tokenisation
->
TF-IDF vectorisation
->
Regression model
->
0–100 bias prediction
->
GUI output

### 1. Article extraction
The program takes a news article URL and extracts the textual content of the page.

### 2. Text preprocessing
The extracted text is cleaned and normalised before analysis, including the removal of punctuation, unnecessary characters and stop words.

### 3. TF-IDF vectorisation
Article language is transformed into numerical features using Term Frequency–Inverse Document Frequency (TF-IDF), weighting terms according to their importance within the dataset.

### 4. Regression
A regression model trained on labelled news articles uses the TF-IDF representation to generate a continuous political-bias prediction.

### 5. User interface
The result is displayed through a GUI alongside information about the currently loaded model and its evaluation statistics.

## Dataset and model

The coursework model was trained using approximately 1,236 labelled news articles**, distributed across left, centre-left, centre, centre-right and right classifications.

Reported model evaluation:

| Metric | Result |
| MAE | 9.78 |
| RMSE | 13.48 |
| R² | 0.87 |

The model outputs predictions on a 0–100 scale rather than assigning articles only to discrete political categories.

## Key challenges

### Quantifying political bias
Political ideology is multidimensional and context-dependent. Reducing it to a single left–right numerical scale is therefore an abstraction rather than an objective measurement.

### Source leakage
An NLP model can accidentally learn the writing style or vocabulary associated with particular publications rather than ideological characteristics themselves. Preventing the model from simply recognising an outlet was therefore an important consideration during development.

### Web scraping
News websites use inconsistent HTML structures, paywalls and anti-bot systems, meaning not every article can be reliably extracted.

### Interpretability
I deliberately used TF-IDF and regression rather than a much larger transformer model. Although less sophisticated, this made the relationship between textual features and model predictions substantially easier to inspect and understand.

## What I learned

The project required me to connect software engineering, natural-language processing and statistical modelling with questions normally treated qualitatively.

The most interesting result was not simply whether the program could generate a bias score, but the limitations encountered when trying to translate concepts such as political language and ideology into numerical variables.

It developed my interest in the intersection between **technology, journalism, politics and quantitative social science**.

## Project documentation

The repository contains the source code and supporting material from my OCR A-Level Computer Science NEA.
