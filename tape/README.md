This is a forked version of the reposatory [https://github.com/songlab-cal/tape](https://github.com/songlab-cal/tape) which contains the code
for running the first set of experiments belonging to the paper "What is a meaningful representation of protein sequences?" This reposatory
contains all of the original code with a few adjustments and some added models and datasets. We refer to the original reposatory for more details.

## Setup

* Clone reposatory
* Run `python setup.py install` for installing the package
Then we could embed it with the UniRep babbler-1900 model like so:

## Data

All the original tape data can be downloaded using the `download_data.sh` script.

For the `beta_lactamase` dataset used in this study, we provide a single `.fasta` file in the `data/beta` folder.

For the unilanguage dataset, we refer to the original authors [reposatory](https://github.com/alrojo/UniLanguage)
where the relevant files can be downloaded by running the `get_data.py` file in the following way:

```python
python get_data.py --domain euk --complete full --quality exp
python get_data.py --domain bac --complete full --quality exp
python get_data.py --domain vir --complete full --quality exp
python get_data.py --domain arc --complete full --quality exp
```
we then provide a script in `data/unilanguage/` called `combine_data.py` for combining the downloaded files into
a single fasta file for training, validation and testing.

## Training and evaluation

For training a model, the two commands `tape-train` and `tape-train-distributed` can be used, where the latter is for in distributed enviroments. 
To train the transformer on masked language modeling, for example, you could run this

```bash
tape-train-distributed transformer masked_language_modeling --batch_size BS --learning_rate LR --fp16 --warmup_steps WS --nproc_per_node NGPU --gradient_accumulation_steps NSTEPS
```
There are a number of features used in training:

    * Distributed training via multiprocessing
    * Half-precision training
    * Gradient accumulation
    * Gradient-allreduce post accumulation
    * Automatic batch by sequence length

The first feature you are likely to need is the `gradient_accumulation_steps`. TAPE specifies a relatively high batch size (1024) by default. This is the batch size that will be used *per backwards pass*. This number will be divided by the number of GPUs as well as the gradient accumulation steps. So with a batch size of 1024, 2 GPUs, and 1 gradient accumulation step, you will do 512 examples per GPU. If you run out of memory (and you likely will), TAPE provides a clear error message and will tell you to increase the gradient accumulation steps.

There are additional features as well that are not talked about here. See `tape-train-distributed --help` for a list of all commands.

### Training a Downstream Model

Training a model on a downstream task can also be done with the `tape-train` command. Simply use the same syntax as with training a language model, adding the flag `--from_pretrained <path_to_your_saved_results>`. To train a pretrained transformer on secondary structure prediction, for example, you would run

```bash
tape-train-distributed transformer stability \
	--from_pretrained results/<path_to_folder> \
	--batch_size BS \
	--learning_rate LR \
	--fp16 \
  	--warmup_steps WS \
  	--nproc_per_node NGPU \
  	--gradient_accumulation_steps NSTEPS \
  	--num_train_epochs NEPOCH \
  	--eval_freq EF \
  	--save_freq SF
```

### Evaluating a Downstream Model

To evaluate your downstream task model, we provide the `tape-eval` command. This command will output your model predictions along with a set of metrics that you specify. At the moment, we support  mean squared error (`mse`), mean absolute error (`mae`), Spearman's rho (`spearmanr`), and accuracy (`accuracy`). Precision @ L/5 will be added shortly.

The syntax for the command is

```bash
tape-eval MODEL TASK TRAINED_MODEL_FOLDER --metrics METRIC1 METRIC2 ...
```

so to evaluate a transformer trained on trained secondary structure, we can run

```bash
tape-eval transformer secondary_structure results/<path_to_trained_model> --metrics accuracy
```

This will report the overall accuracy, and will also dump a `results.pkl` file into the trained model directory for you to analyze however you like.

### List of Models and Tasks

The available models are:

- `transformer`
- `resnet`
- `lstm`
- `unirep`
- `onehot`
- `autoencoder`
- `bottleneck`

The available standard tasks are:

- `language_modeling`
- `masked_language_modeling`
- `remote_homology`
- `fluorescence`
- `stability`
- `beta_lactamase`
- `

The available models and tasks can be found in `tape/datasets.py` and `tape/models/modeling*.py`.
