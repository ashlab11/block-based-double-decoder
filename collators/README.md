These create data collators for all three model types. They are used within the DataLoaders for PyTorch training. The double-decoder pretraining collator creates blocks randomly. Some choices are unnecessary -- i.e. we could just use 8 blocks every time instead of changing the number of blocks we use (log-spaced currently).

TODO: Encoder-decoder collator, both span corruption and prefix-lm (we have to combine them somehow)?
