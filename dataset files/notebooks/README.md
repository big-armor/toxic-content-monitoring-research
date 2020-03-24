# toxic-content-monitoring-research
The collected notebooks pertaining to dataset creation.

Directory:

* Toxic_Sentament.ipynb - A working notebook which contains both the Vader analysis as well as the custom classifier made in NLTK.
* Top_100_Toxic_Words.ipynb - Find top 100 words associated with true examples of each classification problem.
* toxic_rnn.ipynb - notebook for creating an RNN encoder (Seq2Vec) model. I suspect this suffers from over-fitting. Used to create `toxic_rnn.pt` and `toxic_rnn_matrix.out`.
* KaggleWordSplits.ipynb - notebook for text processing based on kaggle conpetition best notebook.
* HF_BERT.ipynb - Used to make encoding set for BERT using the small dataset + the huggingface transformers library.
* allennlp_bert.ipynb - Used to make a BERT encoding using the AllenNLP library.
* Amazon Translation.ipynb - Contains code used to mass-translate russian using Amazon's API.
* basilica_embeddings.ipynb - Notebook that encodes toxic-train-clean.csv using
Basilica and also basilica_y.csv.
