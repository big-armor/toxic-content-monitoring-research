# toxic-content-monitoring-research
The collected datasets made during development.

Directory:

* Depressive data.ods - Original Russian “Depressive and suicidal posts” dataset.
* Depressive translated data.ods - Translated from Russian “Depressive and suicidal posts” dataset.
* toxic_matrix.npz - Sparse matrix representing TFIDF encodings. Used by.
```
x_matrix = scipy.sparse.load_npz('toxic_matrix.npz')
X_train, X_test, y_train, y_test = train_test_split(x_matrix, y, test_size= 0.2)
```
* toxic_suicide_w_RH.zip - Contains toxic_suicide_w_RH.csv. Same as toxic-suicide-train-w-lems.csv, but with additional comments translated from random Russian tweets.
* toxic-suicide-reddit-train-w-lems.zip - Contains toxic-suicide-reddit-train-w-lems.csv. File containing original kaggle dataset fused with text scrapped from reddit.
* toxic-suicide-train-w-lems.zip - Contains toxic-suicide-train-w-lems.csv. File containing original kaggle dataset fused with translated text.
* toxic-train_vader.csv - Sentiment analysis of original dataset, according to VADER.
* toxic-train-w-lems.zip - Contains toxic-train-w-lems.csv. The dataset available through the kaggle competition extended with a column with lemmas derived using Spacy. Lemmatization takes a long time, importing this and using `df[“lemmas”] = df[“lemmas”].apply(eval)` is a lot quicker.
* toxic-train-clean.csv - The dataset available through the kaggle with “comment_text” cleaned.
* toxic-train-clean-small.csv - The dataset available through the kaggle with “comment_text” cleaned, and filtered for a length less than 520.

Other files (most of these are too large to store):
* toxic-train.csv - The dataset available through the kaggle competition.
* toxic_bert_matrix.out - A sliding-window BERT encoding of the text dataset.
* toxic_bert_matrix_small.out - A BERT encoding of the small text dataset (see toxic-train-clean-small.csv).
* toxic_rnn.pt - pretrained Seq2Vec pytorch model. See file “toxic_rnn.ipynb” for usage.
* toxic_rnn_matrix.out - Seq2Vec encodings for cleaned text. Used by;
```
x_matrix = np.loadtxt('toxic_rnn_matrix.out', delimiter=',')
X_train, X_test, y_train, y_test = train_test_split(x_matrix, y, test_size= 0.2)
```
* toxic_elmo_matrix.out - An ELMo encoding of the full text dataset.
