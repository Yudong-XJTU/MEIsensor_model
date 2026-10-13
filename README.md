# MEIsensor_model — MEI family sequence classifier (training code)

Training code, data, and the deployed weights of the deep-learning classifier of
**MEIsensor**, a mobile element insertion (MEI) detection framework for long-read
sequencing data. The classifier assigns each candidate insertion sequence to one of
four classes: **Non-MEI, Alu, LINE1, SVA**.

## Repository contents

```
main.py                     training entry point (argument parsing, seeding, run loop)
model/model.py              VariantSeqClassify: 1-D residual CNN over one-hot DNA sequences
model/seq_classify_model.ph trained weights of the deployed model (state_dict, fp32)
dataset/dataset.py          MEIDataset: CSV -> one-hot tensor pipeline (reads plain or gzipped CSV)
dataset/data/MEI_four_class_train.csv.gz  training set   (30,114 rows; 15,084 unique sequences)
dataset/data/MEI_four_class_test.csv.gz   held-out test set (3,771 rows; all unique)
trainer/trainer.py          TETrainer: training/eval loop, checkpointing
trainer/metrics.py          per-class precision/recall/F1 reporting
trainer/test_training_policy.py          unit tests for the training configuration
```

The train set is stored gzipped because the plain CSV (211 MB) exceeds GitHub's
100 MB single-file limit; `MEIDataset` reads `.csv.gz` transparently (or run
`gunzip dataset/data/*.gz` for plain CSVs).

## Data description

Each CSV row is one candidate insertion: the inserted sequence itself (assembled
from reads; the REF anchor base removed), truncated or right-padded with `N` to
7,000 bp (`N` is encoded as an all-zero one-hot channel). Positives derive from the
HGSVC MEI callset (locus-level; `TE_Designation` mapped to Alu / LINE1 / SVA);
negatives are non-MEI insertion calls. The 8:2 train/test split is per class over
groups of identical sequences (locus-disjoint by construction; zero sequence
overlap between the two sets). LINE1/SVA were resampled with replacement **on the
training side only, after the split**, to class-balance training. All insertion
sites carried by the benchmark trio (HG00512/HG00513/HG00514) are excluded from
both sets.

## Environment

Python >= 3.11 with `torch>=2.0`, `numpy`, `pandas`, `scikit-learn`, `tqdm`
(see `requirements.txt`; `pytest` is optional, for the unit tests). A single
consumer GPU (8–11 GB) is sufficient; `--disable-cudnn` is available if the
cuDNN build misbehaves.

## Train / evaluate

Reproduce training (defaults match the deployed model: 40 epochs, batch 64, Adam
lr 1e-6, seed 42):

```
pip install -r requirements.txt
python main.py            # uses dataset/data/*.csv.gz by default
```

`python main.py --smoke` runs a 1-epoch sanity check. Per-epoch checkpoints
(including optimizer state) are written to `fine_tuned_model/epoch_<N>.ph`; the
trainer reports running loss, accuracy, weighted F1, and a per-class
precision/recall/F1 table on the test set.

To evaluate the deployed weights: load `model/seq_classify_model.ph` (a bare
`state_dict`) into `model.model.VariantSeqClassify`, encode sequences with
`dataset.dataset.MEIDataset`, and take the argmax of the softmax output. On the
provided test set the deployed model reaches **accuracy 0.9873 / weighted-F1
0.9873 / macro-F1 0.9838** (per-class F1: Non-MEI 0.985, Alu 0.992, LINE1 0.969,
SVA 0.989).

## Citation

If you use this code, please cite the MEIsensor paper (citation to be added upon
publication).

## License

This project is released under the [MIT License](LICENSE).
