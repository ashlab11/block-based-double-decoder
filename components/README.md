Components folder includes the code for all the subparts of the models. 

1. Attention folder includes three attention types:
   1. Self attention (normal attention for decoder and our encoder)
   2. Combo attention (combines both within-block self attention and across-block full attention)
   3. Cross attention (standard cross-attention for the encoder-decoder baseline)
Note here that combo attention does the two attentions in **parallel** -- rather than doing cross then adding self or vice versa, it does it at the same time and weighs it according to the log-sum-exp so it's essentially just doing one single attention but with two different key-value matrix pairs (self and cross, sometimes shared if `shared = True`).

2. Block masks includes the code for directly **creating** the block masks, including all the torch compile. Tough to look at, sorry. 
3. Layers includes the code for full layers, from which the entire models are created. Three types -- causal (self attention), combo attention, and standard (self + cross attention), which together create decoder-only, double-decoder, and standard encoder-decoder models respectively.
4. 