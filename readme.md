## LMG

<div align=center><img src="assets/overview.jpg"></div>

## Credits
This repository is build upon [*SCAN*](https://github.com/kuanghuei/SCAN) (Lee et al.) and [*NAAF*](https://github.com/CrossmodalGroup/NAAF) (Zhang et al.).
We express our sincere gratitude to the researchers for providing their code, which has been instrumental in the
development of this project.

## Environment Configuration
```shell
conda create -n lmg python=3.7
pip install -r requirements.txt
```
```shell
python -c "import nltk; nltk.download('all')"
```

## Data Preparation
<table>
	<tr>
	    <th align="center">Dataset</th>
	    <th align="center">File</th>
	    <th align="center">Source</th>  
	</tr >
	<tr >
	    <td rowspan="8" align="center">Flickr30K</td>
	    <td align="center">train_ims.npy</td>
	    <td rowspan="6" align="center"><a href="https://www.kaggle.com/datasets/kuanghueilee/scan-features">SCAN</a>
 (Lee et al.)</td>
	</tr>
	<tr>
	    <td align="center">train_caps.txt</td>
	</tr>
	<tr>
	    <td align="center">dev_ims.npy</td>
	</tr>
	<tr>
	    <td align="center">dev_caps.txt</td>
	</tr>
    <tr>
        <td align="center">test_ims.npy</td>
	</tr>
	<tr>
        <td align="center">test_caps.txt</td>
	</tr>
    <tr>
	    <td align="center">glove_840B_f30k_precomp.json.pkl</td>
        <td align="center"><a href="https://github.com/CrossmodalGroup/NAAF/blob/main/vocab/glove_840B_f30k_precomp.json.pkl">NAAF</a>  (Zhang et al.)</td>
	</tr>
	<tr>
	    <td align="center">glove_f30k_word_idx.json</td>
        <td align="center">data/vocab_word_idx/glove_f30k_word_idx.json</td>
	</tr>
    <tr >
	    <td rowspan="10" align="center">MSCOCO</td>
	    <td align="center">train_ims.npy</td>
	    <td rowspan="8" align="center"><a href="https://www.kaggle.com/datasets/kuanghueilee/scan-features">SCAN</a>
 (Lee et al.)</td>
	</tr>
	<tr>
	    <td align="center">train_caps.txt</td>
	</tr>
	<tr>
	    <td align="center">dev_ims.npy</td>
	</tr>
	<tr>
	    <td align="center">dev_caps.txt</td>
	</tr>
    <tr>
        <td align="center">test_ims.npy</td>
	</tr>
	<tr>
        <td align="center">test_caps.txt</td>
	</tr>
    <tr>
        <td align="center">testall_ims.npy</td>
	</tr>
	<tr>
        <td align="center">testall_caps.txt</td>
	</tr>
    <tr>
	    <td align="center">glove_840B_coco_precomp.json.pkl</td>
        <td align="center"><a href="https://github.com/CrossmodalGroup/NAAF/blob/main/vocab/glove_840B_coco_precomp.json.pkl">NAAF</a>  (Zhang et al.)</td>
	</tr>
	<tr>
	    <td align="center">glove_coco_word_idx.json</td>
        <td align="center">data/vocab_word_idx/glove_coco_word_idx.json</td>
	</tr>
</table>
The provided image features lack information related to bounding boxes, making visualization impossible. To obtain such information, it is necessary to re-extract fine-grained 2048-dimensional features. You can refer to https://github.com/peteanderson80/bottom-up-attention and https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome for guidance on this task.

## Train

```shell
bash -i scripts/train.sh
```
We use a single RTX 3090 to train the model, with a batch size set to 256 for both Flickr30K and MSCOCO datasets.

## Evaluate
```shell
bash -i scripts/test.sh
```
The training logs and checkpoints will be saved in directory exps/exp_name.
