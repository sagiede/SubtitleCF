# SubtitleCF
Exploring the Use of Subtitles for Enhanced Movie Recommendations - Git repo for RecSys paper.

Please note that the SubLens ml-20m Dataset will be released upon acceptance.

Therefore, Some of the functions of this repo are limited. 

### Installations
please make sure you satisfy all of the packages requirements on requirements.txt file 

### Notes
In the repo there are prerpocessed files including:

- preprocessed json ready for experiment.
- pretrained models for testing.
- si vectors for training new models.
- LDA gensim model for plotting movies topics.

please run Main.py -h for the full explanations and available models.

For Running the Code run in the cmd Main.py, and add arguments for the specific task.


#### Plot movie topics by id
plot_movie_topics.py -r plot_movie_topics --movie <movie_imdb_id> --topics <n_topics_to_print> --words <n_words_in_topic>
imdb id can be retrieved from the address line (the number after the tt). 
We provided a screen shot in extract_imdb_id.jpg to demonstrate that.


#### Training:
Main.py -r train_rs --rsname <model_name> 

#### Evaluating model:
Main.py -r test_rs --rsname <model_name>



### pre-process dataset
Main.py -r preprocess_data

## Upon acceptance methods
The next methods require the Subtitles, Hence, will be available upon acceptance
methods: create_si_vectors, create_sentences_bert_embeddings, train_cnn_model

