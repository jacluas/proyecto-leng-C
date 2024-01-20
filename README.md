# Mexico 120 Flower Model
### Pre-trained 512-VGG model (from Mexico 120 Flower dataset) Modified

We share the pre-trained 512-VGG CNN model as research support in Mexico native flora identification. We believe that this pre-trained model may be a good starting point for future research in flower identification. Transfer learning from pre-trained models on ImageNet 2012 dataset was used.

Pre-trained model can be found here: [https://github.com/jacluas/Mexico120FlowerModel/releases](https://github.com/jacluas/Mexico120FlowerModel/releases)


### Usage

The model is shared in the form of H5 format, that is, a file format to store structured data to easily share. Keras saves models in this format as it can easily store the weights and model configuration in a single file. 


### How to consume the pre-trained 512-VGG model (from Mexico 120 Flower dataset)

We used Keras version 2.2.4, with Tensorflow 1.13.1 as backend, and Python version 3.7.3.

You can evaluate a sample image by performing the following:

```python
python predict.py MODEL_NAME.h5 IMAGE_TEST_PATCH TOP-K
```

Examples Top-_1_:
```python
	python predict.py model/model_512_vgg TEST/Achillea_millefolium/AM1.jpeg

	Predictions:
	'Achillea millefolium',	0.9999955892562866
```
```python
	python predict.py mode/model_512_vgg TEST/Cordia_boissieri/CB1.jpeg -k 1
	
	Predictions:
	'Cordia boissieri', 1.0
```

Examples Top-_5_:
```python

	python predict.py model/model_512_vgg TEST/Achillea_millefolium/AM1.jpeg -k 5
	
	Predictions:
	'Achillea millefolium',		0.9999955892562866
	'Heliotropium angiospermum',	4.443768830242334e-06
	'Turnera diffusa',		1.945797342695066e-11
	'Asclepias subulata',		1.160856393650489e-11
	'Verbesina encelioides',	9.17614335210759e-12
```
```python
	python predict.py model/model_512_vgg TEST/Cordia_boissieri/CB1.jpeg -k 5
	
	Predictions:
	'Cordia boissieri',		1.0
	'Pontederia crassipes',		6.602185909088121e-09
	'Argemone mexicana',		7.986839661855427e-11
	'Oenothera speciosa',		4.6942175840891665e-11
	'Cosmos bipinnatus',		3.5377999358168766e-13

```
