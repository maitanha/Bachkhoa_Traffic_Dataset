***Velocity Prediction Model using ResNet18***

This model is inspired by PyTorch's ResNet18 
<a>https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html</a>

In order to run this file, in case you have installed all the imported libraries in your environment. Just run like normal using <code>Makefile</code> described below. 

Or else, check out how to use <code>venv</code>:
<a>https://sparkbyexamples.com/python/python-activate-virtual-environment-venv/</a>

All things considered, you can install the necessary packages within this environment or add image to <b>data</b> folder.

How the <code>data</code> folder is structured:
```
Working Directory/
├── data/
│   ├── full/
│   │   ├── train/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.png
│   │   │   ├── ...
│   │   └── val/
│   │       ├── image1.jpg
│   │       ├── image2.png
│   │       ├── ...
│   └── ...
├── ...
└── README.MD
```
**Important note:** The used folder must contain two subfolders, whose names **must** be **train** and **val**.

Now, you are ready to go. I have implemented a simple interface for ease of use. You only need to pass in the parameters... or maybe not at all :D 

**Model training**
Run <code>make train</code> to start training the model
This command will require you to pass in:
<ul>
    <li>Number of epochs. Ex: <b>20</b></li>
    <li>A name for the model after training (should have a *.pth extension). Ex: <b>test_model.pth</b></li>
    <li>Data path. Ex: <b>./data/full</b></li>
</ul>

**Model Validation**
Run <code>make validate</code> to start validate / test your model
This command will require you to pass in:
<ul>
    <li>The trained model name (that you train above). Ex: <b>test_model.pth</b></li>
    <li>Data path. Ex: <b>./data/full</b></li>
</ul>

**All in the nutshell**
Run <code>make all</code> for ease of use :D