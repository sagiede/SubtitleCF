# SubtitleCF
##### Exploring the Use of Subtitles for Enhanced Movie Recommendations - Git repo for RecSys paper.

SubLens ml-20m Dataset will be released upon acceptance.
 

#### Installations
The Code was built using conda and Python 3.8.
Make sure you satisfy all of the requirements of the packages on the requirements.txt

Before running the scripts, Access the next link:
https://drive.google.com/file/d/1TWaxJ0OrLi-dZundNr2xL4SXvxGfjGQs
 
 and extract the DATA folder into SubtitleCF.

### Notes
In the repo, there are pre-processed files including:

- Preprocessed JSON ready for experiments.
- Pretrained models for testing.
- Si vectors for training new models.
- LDA gensim model for plotting movie topics.

Run "python Main.py -h" for the full explanations and available models for the RS scripts.


### Plot movie topics by id (tag clouds)
Given a movie, this script will print its most dominant topics by their dominance order (The first topic is the most representative).

Command: python plot_movie_topics.py --movie <movie_imdb_id> --topics <n_topics_to_print> --words <n_words_in_topic>

for example: python plot_movie_topics.py --movie 468569 --topics 8 --words 30 

IMDB id can be retrieved from the address line (the number after the tt). 
extract_imdb_id.jpg shows how to get the IMDB id of a movie.

Run "python plot_movie_topics.py -h" for additional examples 

### Training:
Command:  python Main.py -r train_rs --rsname <model_name> 

run "python Main.py -h" for available model names

### Evaluating model:
Command:  python Main.py -r test_rs --rsname <model_name>

run "python Main.py -h" for available model names

### Pre-process dataset
Command:  python Main.py -r preprocess_data

### Upon acceptance methods
The next methods require Subtitles. Hence, they will be available upon acceptance.

- create_si_vectors
- create_sentences_bert_embeddings
- train_cnn_model

