<h1 align="center">Attention Is All You Need</h1>
PyTorch implementation of the transformer architecture presented in "Attention Is All You Need" 
by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, 
Lukasz Kaiser, Illia Polosukhin


## Architecture
<div align="center">
  <img src="docs/architecture.png" width="600px" />
</div>


### Positional Encoding
The position of each token in a sequence is encoded using the following formula and then
added on top of the token's embedding vector.

<div align="center">
  <img src="docs/positional_encoding_formula.png" width="300px" />
</div>

<div align="center">
  <img src="docs/positional_encoding.png" width="600px" />
</div>


### Multi-head Attention
In a multi-head attention sublayer, the input queries, keys, and values are each projected into 
`num_heads` vectors of size `d_model / num_heads`. Then, `num_heads` scaled dot-product
attention operations are performed in parallel, and their outputs are concatenated and projected back into 
size `d_model`.


## Methods
Overall, this implementation almost exactly follows the architecture and parameters described in [1]. 
However, due to limited resources, I instead trained using the smaller `Multi30k` machine translation dataset.


### Learning Rate Schedule
The learning rate schedule used in [1] is shown below:

<div align="center">
  <img src="docs/paper_lr_schedule_formula.png" width="500px" />
</div>

<div align="center">
  <img src="docs/paper_lr_schedule.png" width="600px" />
</div>

However, during my experiments, models trained using this schedule failed to achieve BLEU scores above `0.01`.
Instead, I used PyTorch's `ReduceLROnPlateau` scheduler, which decreases the learning rate by `factor=0.5` every
time the validation loss plateaus:

<div align="center">
  <img src="experiments/en-de/08_19_2023/17_27_32/lr.png" width="600px" />
</div>


## Results


## Notes
- Input of size `batch_size * sequence_length * embedding_size`
- length of sequence can vary: `W_Q, W_K, W_V` are all of shape `d_model * d_v` which does not depend on sequence length
- output of the topmost encoder layer is fed in as the `key` and `value` into each layer in the decoder
- "Teacher forcing"
  - during training, the transformer has access to the ground truth words before the current position
  - It then generates a prediction using those words at the current position
  - But what about the first word? How does transformer predict first word without previous decoder outputs?
    - Special "start of sequence" symbol `<sos>` is at the start of every sequence, fixes this off-by-one issue
    - On first pass, transformer sees the `source` sequence and `prev=[<sos>]`, so it knows to predict the first word
    - On second pass, transformer now sees `source` and `prev=[<sos>, first_prediction]`


# Setup Instructions
1. Run `python -m pip install -r requirements.txt`
2. Download spacy language pipelines
```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## References

[[1](https://arxiv.org/abs/1706.03762)] 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Attention Is All You Need. 
_arXiv:1706.03762 [cs.CL]_
