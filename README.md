# SubtitleCF
Exploring the Use of Subtitles for Enhanced Movie Recommendations - Git repo for RecSys paper.

Please note that the SubLens ml-20m Dataset will be released upon acceptance.

Therefore, Some of the functions of this repo are limited. 

### Installations
The Code was built using Python 3.8.
Please make sure you satisfy all of the packages requirements on requirements.txt file 

### Notes
In the repo there are pre-processed files including:

- Preprocessed json ready for experiment.
- Pretrained models for testing.
- Si vectors for training new models.
- LDA gensim model for plotting movies topics.

Please run Main.py -h for the full explanations and available models.

For Running the Code run in the cmd Main.py, and add arguments for the specific task.


#### Plot movie topics by id
Plot_movie_topics.py -r plot_movie_topics --movie <movie_imdb_id> --topics <n_topics_to_print> --words <n_words_in_topic>
Imdb id can be retrieved from the address line (the number after the tt). 
We provided a screen shot in extract_imdb_id.jpg to demonstrate that.


#### Training:
Main.py -r train_rs --rsname <model_name> 

#### Evaluating model:
Main.py -r test_rs --rsname <model_name>



### pre-process dataset
Main.py -r preprocess_data

## Upon acceptance methods
The next methods require the Subtitles, Hence, will be available upon acceptance
Methods: create_si_vectors, create_sentences_bert_embeddings, train_cnn_model

