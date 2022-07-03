# SubtitleCF
##### Investigating the Value of Subtitles for Improved Movie Recommendations - Git repo for UMAP 2022: 30th ACM Conference.
 

## Sublens-20M Dataset
https://tinyurl.com/sublens-20m

#### Properties:
The dataset contains 19.25k subtitles for movies from Movielens-20M Dataset.
There are 2 folders:
- subtitles: the subtitles files - each file name is the imdbID of the movie
- Sublens_20M_metadata: the links, movies csv files from movielens-20M (the original files and filtered ones by the subtitles exists)

In overall, the dataset matches 71% of the movies, which covers 98.3% of the interactions in movielens-20M Dataset.
Each file of movie subtitle contains in avg 7.3k words and 1.6k sentences. 

## SubtitleCF Implementation

##### Installations
The Code was built using conda and Python 3.8.
Make sure you satisfy all of the requirements of the packages on the requirements.txt

Before running the scripts, Access the next link:
https://drive.google.com/file/d/1TWaxJ0OrLi-dZundNr2xL4SXvxGfjGQs
 
 and extract the DATA folder into SubtitleCF.

#### Notes
In the repo, there are pre-processed files including:

- Preprocessed JSON ready for experiments.
- Pretrained models for testing.
- Si vectors for training new models.
- LDA gensim model for plotting movie topics.

Run "python Main.py -h" for the full explanations and available models for the RS scripts.


#### Plot movie topics by id (tag clouds)
Given a movie, this script will print its most dominant topics by their dominance order (The first topic is the most representative).

Command: python plot_movie_topics.py --movie <movie_imdb_id> --topics <n_topics_to_print> --words <n_words_in_topic>

for example: python plot_movie_topics.py --movie 468569 --topics 8 --words 30 

IMDB id can be retrieved from the address line (the number after the tt). 
extract_imdb_id.jpg shows how to get the IMDB id of a movie.

Run "python plot_movie_topics.py -h" for additional examples 

#### Extract SI Embeddings:
Extracting the Component 1 SI vectors (according to the requested model)
Command:  python Main.py -r create_si_vectors --rsname <model_name> 

#### Training:
Train the Integrated model using the SI vectors (or without in Vanilla BPR case)
Command:  python Main.py -r train_rs --rsname <model_name> 

run "python Main.py -h" for available model names

#### Evaluating model:
Command:  python Main.py -r test_rs --rsname <model_name>

run "python Main.py -h" for available model names

#### Exctract BERT Sentences vectors:
Command:  python Main.py -r create_sentences_bert_embeddings

#### Pre-process dataset
Command:  python Main.py -r preprocess_data


## Citation

- Sagi Eden. (2022). Investigating the Value of Subtitles for Improved Movie Recommendations.
- https://dl.acm.org/doi/10.1145/3503252.3531291
- https://github.com/sagiede/SubtitleCF

If you find this code/Dataset useful in your research, please cite the following paper:

Sagi Eden, Amit Livne, Oren Sar Shalom, Bracha Shapira, and Dietmar Jannach. 2022. Investigating the Value of Subtitles for Improved
Movie Recommendations. In UMAP ’22: ACM Conference on User Modeling, Adaptation and Personalization, July 4th-7th, 2022, Barcelona,
Spain. ACM, New York, NY, USA, 17 pages.

BibTeX:
```bibtex
@inproceedings{10.1145/3503252.3531291,
author = {Eden, Sagi and Livne, Amit and Sar Shalom, Oren and Shapira, Bracha and Jannach, Dietmar},
title = {Investigating the Value of Subtitles for Improved Movie Recommendations},
year = {2022},
booktitle = {Proceedings of the 30th ACM Conference on User Modeling, Adaptation and Personalization},
pages = {99–109},
}
```

For the original Dataset of Movielens-20M:
https://grouplens.org/datasets/
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872


