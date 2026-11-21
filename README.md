# Climate Change Discussions on Reddit

An end-to-end social media analytics project examining how climate change is discussed across Reddit communities. The project combines data acquisition, data engineering, natural language processing, topic modeling, temporal analysis, emotion detection, and machine-learning-based engagement prediction.

The analysis uses **9,948 top posts** collected from **10 climate-related subreddits**, covering posts dated from **June 2012 to June 2025**. In addition to each post's title and body, the highest-ranked top-level comment was included to capture part of the community's response.

## Project objectives

This project was designed to answer the following questions:

- How do sentiments differ across climate-related Reddit communities?
- Which emotions and discussion themes appear most frequently?
- How have climate-related topics and engagement changed over time?
- How do rule-based and transformer-based sentiment models compare?
- Which textual and community-level factors help predict post engagement?

## Data sources

Data was collected through the Reddit API using [PRAW](https://praw.readthedocs.io/). Up to 1,000 top posts were requested from each of the following subreddits:

- `r/climatechange`
- `r/environment`
- `r/climateskeptics`
- `r/climate`
- `r/sustainability`
- `r/EcoInternet`
- `r/Collapse`
- `r/energy`
- `r/renewableenergy`
- `r/ZeroWaste`

The collected fields included the subreddit, post ID, title, body text, score, number of comments, creation time, author, URL, highest-ranked top-level comment, its author, and its score. Collection was performed separately for each subreddit so interrupted API runs could resume without downloading completed communities again. The resulting files were then combined into a single dataset.

### Dataset summary

| Item | Value |
|---|---:|
| Total records | 9,948 |
| Subreddits | 10 |
| Duplicate rows | 0 |
| Link posts | 9,445 |
| Self-posts | 503 |
| Earliest post | 26 June 2012 |
| Latest post | 8 June 2025 |
| Average combined text length | 68.5 words |

The dataset contains top posts rather than a random sample of all Reddit content. The analysis therefore represents highly visible climate discussions and should not be interpreted as a complete picture of Reddit users or public opinion.

## Methodology

### 1. Data acquisition and engineering

The Reddit API workflow includes environment-based credential loading, light request throttling, API exception handling, progress reporting, and per-subreddit CSV output.

During preprocessing:

- Unix timestamps were converted into readable dates.
- Missing values and duplicate rows were examined.
- Post titles, body text, and the highest-ranked top-level comment were combined into one text field.
- URLs embedded in the combined text were removed with a regular expression.
- Language detection was used to verify that the analyzed text was English.
- User-related fields and fields not required for analysis were removed from the final dataset.
- Text length was calculated as the number of words in each combined document.

The analysis also distinguished link posts from self-posts. Link posts had a higher mean Reddit score, while self-posts generated substantially more comments on average.

### 2. Engagement feature engineering

A custom engagement score was developed from three Reddit interaction measures:

- Post score as a popularity measure
- Number of comments as a discussion measure
- Top-comment score as a response-depth measure

The score was calculated as:

```text
engagement_score = log(1 + post_score + 3 × num_comments + 2 × top_comment_score)
```

The logarithmic transformation reduces the effect of extremely popular posts. Missing top-comment scores were filled with the column mean. For descriptive analysis, engagement was divided into three groups:

- Low: below `4.0`
- Medium: `4.0` to `7.8`
- High: above `7.8`

### 3. Sentiment analysis

Two sentiment methods were applied and compared:

1. **VADER** — a lexicon- and rule-based model that produces a compound sentiment score.
2. **Twitter-RoBERTa** — the Hugging Face model `cardiffnlp/twitter-roberta-base-sentiment`, used to classify text as negative, neutral, or positive.

The transformer classified the records as follows:

| Sentiment | Number of records |
|---|---:|
| Negative | 4,994 |
| Neutral | 2,855 |
| Positive | 2,099 |

The models sometimes produced opposing labels. These disagreements were examined manually, showing how contextual phrasing, mixed sentiment, and climate-related language can be interpreted differently by rule-based and transformer models.

Across communities, `r/climateskeptics` and `r/Collapse` contained comparatively more negative classifications, while `r/ZeroWaste` contained more positive discussion. Negative posts were also common in both the medium- and high-engagement groups, indicating that engagement does not increase linearly with positive sentiment.

### 4. Emotion detection and phrase analysis

Multi-label emotion detection was performed with `SamLowe/roberta-base-go_emotions`. Labels with a confidence above `0.30` were retained. Neutral was the most frequent emotion, followed by emotions such as curiosity, approval, admiration, and optimism.

Bigram analysis was used to investigate the language associated with different emotions. Frequent meaningful phrases included:

- `fossil fuel`
- `clean energy`
- `renewable energy`
- `climate change`

The analysis also identified Reddit-generated boilerplate such as `submission statement` and `reply op`, which contributed to neutral classifications and represents an opportunity for stronger preprocessing in future work.

### 5. Temporal and topic analysis

Annual post counts, average engagement, and transformer confidence scores were explored over time. The number of sampled climate-related posts increased substantially after 2016. The largest annual count in the dataset was **1,707 posts in 2024**; the 2025 count covers only a partial year.

Two topic-modeling approaches were used:

- **BERTopic** to identify the most prominent topic within individual years
- **Latent Dirichlet Allocation (LDA)** with bigram features to identify ten broader themes across the full corpus

The major topics were interpreted as:

1. Zero waste and plant-based living
2. Climate-change effects
3. Fossil fuels
4. Renewable energy and technology
5. Meta/moderation messages
6. Trump and the Green New Deal
7. Energy transition
8. Clean energy and sustainability
9. US politics
10. Metadata and miscellaneous content

Prominent annual themes included environmental policy and Trump in 2017, coal and renewable energy in 2018, plastic waste from 2019 onward, global warming and extreme heat in 2023, and solar and wind power in 2024.

### 6. Engagement prediction

The project predicts the engineered engagement score using:

- TF-IDF features from the combined text, limited to 1,000 terms
- Text length
- Subreddit, one-hot encoded
- VADER sentiment score
- Transformer sentiment confidence

The data was split into 80% training and 20% testing sets. A scikit-learn pipeline handled feature transformation and model training.

#### Random Forest

The Random Forest regressor produced:

| Metric | Result |
|---|---:|
| Mean squared error (MSE) | 0.34 |
| Mean absolute error (MAE) | 0.41 |
| R² | 0.80 |

> The notebook output labels `0.34` as RMSE, but the code calls `mean_squared_error` without taking its square root. It is therefore reported here as MSE.

Feature importance indicated that the subreddit in which a post appeared was more influential than its sentiment score. Text length also contributed to engagement prediction.

#### XGBoost

XGBoost was tuned with 3-fold `GridSearchCV` across 48 training fits. The best parameters were:

```text
learning_rate = 0.1
max_depth = 3
n_estimators = 200
subsample = 0.8
```

The best cross-validation RMSE was approximately `0.565`. As with the Random Forest, community membership dominated the most important features, suggesting that differences in subreddit size, activity, and audience affect engagement considerably.

## Main findings

- Climate-related Reddit discussion varies strongly by community.
- Negative sentiment was the most frequent transformer classification in this dataset.
- High engagement was not limited to positive content; negative posts also attracted substantial interaction.
- Neutral language was common, partly because of factual headlines and Reddit-specific boilerplate.
- Fossil fuels, renewable energy, plastic waste, sustainability, climate impacts, and US politics were recurring themes.
- Climate discussion gained substantial traction after 2016 in this top-post dataset.
- Subreddit identity and text length were more useful for predicting engagement than sentiment confidence alone.

## Repository contents

```text
SMA_capstone/
├── Data acqusition and engineering.ipynb
├── Sentiment analysis and prediction.ipynb
├── climatechange_combined.csv
├── final.csv
├── climatechangereport_binayak.pdf
├── output.png
└── README.md
```

- **Data acquisition and engineering notebook** — Reddit API collection, dataset combination, exploratory analysis, cleaning, and engagement feature engineering.
- **Sentiment analysis and prediction notebook** — sentiment and emotion models, temporal analysis, topic modeling, visualizations, and engagement prediction.
- **`climatechange_combined.csv`** — combined raw data from the ten subreddits.
- **`final.csv`** — cleaned and engineered dataset used in the analysis notebook.
- **PDF report** — project report containing the analysis and conclusions.

## Running the project

### Prerequisites

- Python 3.10 or later
- Jupyter Notebook or JupyterLab
- Reddit API credentials, required only to collect the data again

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn praw python-dotenv langdetect nltk transformers torch tqdm scikit-learn bertopic wordcloud xgboost
```

Download the required NLTK resources:

```python
import nltk

nltk.download("stopwords")
nltk.download("vader_lexicon")
```

To recollect Reddit data, create a `.env` file in the project directory:

```env
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
USER_AGENT=your_application_user_agent
```

Do not commit API credentials to version control.

Run the notebooks in this order:

1. `Data acqusition and engineering.ipynb`
2. `Sentiment analysis and prediction.ipynb`

The provided `final.csv` allows the second notebook to be explored without repeating Reddit collection.

## Limitations

- Reddit's top-post endpoint favors popular content and introduces selection bias.
- The subreddits differ in membership, posting frequency, moderation, and culture, so raw engagement is not directly comparable across communities.
- Only the highest-ranked top-level comment was included; the full discussion threads were not analyzed.
- Combining a post with its top comment creates a general discussion representation but can mix the author's view with the community's reaction.
- The transformer models were not fine-tuned or manually validated on a labeled climate-change dataset.
- Classification confidence is not the same as a continuous positive-to-negative sentiment score.
- Observed temporal relationships, including changes around the COVID-19 period, are associations and do not establish causation.
- The 2025 data represents only a partial year.

## Future improvements

- Remove Reddit-specific boilerplate before sentiment and emotion classification.
- Manually label a representative validation sample and compare model accuracy.
- Use climate-domain embeddings or fine-tuned transformer models.
- Normalize engagement for subreddit size, post age, and community activity.
- Use time-aware train/test splitting to evaluate performance on future posts.
- Apply SHAP values for more interpretable model explanations.
- Analyze complete comment threads and separate post sentiment from response sentiment.

## Responsible use

This project analyzes public Reddit discussions for educational and research purposes. Usernames and comment-author identifiers were excluded from the final analytical dataset. Any reuse should follow Reddit's terms, API policies, and appropriate privacy and research-ethics practices.
