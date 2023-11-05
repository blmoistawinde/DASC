# DASC

Code and data for the EMNLP 2023 paper ["Semantic Space Grounded Weighted Decoding for Multi-Attribute Controllable Dialog Generation"](paper.pdf).

## Train

To train DASC model on dulemon:

```
bash dulemon_scripts/run_dulemon_dasc1.sh
```

Similar goes to other baselines with `run_dulemon_*.sh`

To train DASC model on esconv:

```
bash esconv_scripts/run_dasc1.sh
```

## Generate

Using the `*_topp.ipynb` in the `dulemon_scripts` or `esconv_scripts`

## Evaluate

Using the `eval_*.ipynb` in the `dulemon_scripts` or `esconv_scripts`