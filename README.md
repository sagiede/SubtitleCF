# SubtitleCF
##### Exploring the Use of Subtitles for Enhanced Movie Recommendations - Git repo for RecSys paper.

SubLens ml-20m Dataset will be released upon acceptance.
 

#### Installations
The Code was built using Python 3.8.
Make sure you satisfy all of the packages requirements on requirements.txt file 

Before running the scripts, Access the next link:
https://drive.google.com/file/d/1a2DsLHm34Lbo9eE8olyzFgKQYGM5wJ2E
 
 and extract the DATA folder into SubtitleCF.

### Notes
In the repo there are pre-processed files including:

- Preprocessed json ready for experiment.
- Pretrained models for testing.
- Si vectors for training new models.
- LDA gensim model for plotting movies topics.

Run "python Main.py -h" for the full explanations and available models.


### Plot movie topics by id (tag clouds)
python plot_movie_topics.py --movie <movie_imdb_id> --topics <n_topics_to_print> --words <n_words_in_topic>

for example: python plot_movie_topics.py --movie 468569 --topics 8 --words 30 

Imdb id can be retrieved from the address line (the number after the tt). 
We provided a screen shot in extract_imdb_id.jpg to demonstrate that.

### Training:
python Main.py -r train_rs --rsname <model_name> 

run "python Main.py -h" for available model names

### Evaluating model:
python Main.py -r test_rs --rsname <model_name>

run "python Main.py -h" for available model names

### Pre-process dataset
python Main.py -r preprocess_data

### Upon acceptance methods
The next methods require the Subtitles, Hence, will be available upon acceptance.

- create_si_vectors
- create_sentences_bert_embeddings
- train_cnn_model

