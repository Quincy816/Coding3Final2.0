# Flower Identification Project Using CNN & PyQt5 - The Language of Flowers
Name: QIAN XU

Student ID: 22018776

Third-party code resources Link: [https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test6_mobilenet](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/Test6_mobilenet)

Tutorial Link: [https://www.bilibili.com/video/BV1yE411p7L7/?vd_source=84bad5c2512fc2824e0af4a786add6a8](https://www.bilibili.com/video/BV1yE411p7L7/?vd_source=84bad5c2512fc2824e0af4a786add6a8)

Dataset Link: [https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data)

The idea of this project can be divided into several main steps: data preparation and pre-processing, model training, model deployment and application development. A detailed description of each step is given below:

## 1. Data preparation and pre-processing
### Collecting the dataset: 
In this phase, it was started by collecting the dataset of pictures of different kinds of flowers. The dataset was obtained from: 
[https://www.kaggle.com/datasets/alxmamaev/flowers-recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

### Data partitioning: 
The dataset is divided into 8:2 training, validation and test sets. The training set is used for model learning, the validation set is used for tuning hyperparameters and preventing overfitting, and the test set is used for final performance evaluation.
Code to handle dataset references from 
[https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/data_set/split_data.py](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/data_set/split_data.py)

The third-party code snippets used for reference are as follows:
```
def main():
...
random.seed(0)
    split_rate = 0.1
    ...
        for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

print("processing done!")
```


### Dataset preprocessing: 
PyTorch's ImageFolder class is used to load the image dataset and perform preprocessing operations on the images. The path where the dataset is located is root_dir (flower dataset), and the loaded image has gone through a series of preprocessing operations:
Resize the image to 224x224 pixels (transforms.Resize((224, 224))).
Convert the image to a PyTorch tensor (transforms.ToTensor()).
Normalise the image (normalise using ImageNet's mean and standard deviation) (transforms.Normalize()).
Data preprocessing code adapted from: 
[https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/custom_dataset/main.py](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/custom_dataset/main.py)

The third-party code snippets used for reference are as follows:
```
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)
```

![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/3e68cb88-e979-47f6-a8cf-1afc89c01e2a)


## 2. Model Training
Choosing a model architecture: choose a deep learning model such as Convolutional Neural Network (CNN) that is suitable for the image classification task. Existing architectures can be used such as ResNet, Inception or VGG to name a few.

### Pre-trained Model MobileNetV2: 
The pre-trained model MobileNetV2 was used in this project. In the function build_model, models.mobilenet_v2(pretrained=pretrained) is used to load the pre-trained MobileNetV2 model when the parameter pretrained=True. MobileNetV2 is a lightweight convolutional neural network architecture for resource-constrained environments such as mobile and embedded devices. The model is pre-trained on the ImageNet dataset and provides decent image feature extraction.

![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/15fc8668-2f48-4e41-931b-779fb4569ae2)

At the end of the function, the number of output categories is modified to 5 by modifying the last fully-connected layer (model.fc = nn.Linear(1024, 5)) to be used for the current classification task where the flower category is 5. (The exact code can be found in "model.py")

### Training the model: 
The training set data is used to train the model. This process involves forward propagation, loss calculation, backpropagation and weight update.
![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/004f38cb-2695-418d-949a-072d3decd5ff)

model.train() instructs the model to enter training mode.
For each training batch, it performs forward propagation (model(image)), computes the loss (criterion(outputs, labels)), and performs the steps of backpropagation (loss.backward()) as well as weight update (optimiser.step()).
train_running_loss and train_running_correct are used to accumulate the number of losses and correct predictions.
Finally, it calculates the average loss and accuracy for each epoch and returns these values.

### Hyperparameter tuning: 
Testing the performance of the model on the validation set and tuning hyperparameters such as learning rate, batch size, number and size of model layers.
![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/4ef864fc-c488-4cbf-8bec-15ddf2e26b47)

### Model Evaluation: 
Evaluating the accuracy and other performance metrics of the model on a test set to ensure that the model has good generalisation capabilities.
![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/411ac033-3f7a-45f1-a8e1-22d6ec822e5c)
(Check "train.py" for the exact code)

## 3. Model deployment
### Model Save:
After training, the model is saved as a file for use in the application, and line plots of train_acc, valid_acc, train_loss, valid_loss are generated at the same time.

<img width="640" alt="image" src="https://github.com/Quincy816/Coding3Final2.0/assets/115622644/8ff63a7d-8cea-468a-9113-be660fafb63a">

### Create a prediction function:
Write a function (e.g. main1) that takes an input image, preprocesses it, and then uses the trained model to make a prediction that returns the type of flower. Here I have quoted part of the code from github, 
link: 
[https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/predict.py](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test6_mobilenet/predict.py)

The third-party code snippets used for reference are as follows:
```
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = MobileNetV2(num_classes=5).to(device)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()
```

<img width="538" alt="image" src="https://github.com/Quincy816/Coding3Final2.0/assets/115622644/fd17bb76-ee2a-4241-afad-f8b6dd8028b4">

(Check "predict.py" for the exact code.)

## 4. Application development
Designing the User Interface: design a graphical user interface using Qt and PyQt5. The interface should include components for selecting pictures, displaying pictures, displaying prediction results and floral language.
Integrate Model: Integrate the previously saved model into the application. When the user selects an image and clicks "Run", the application should call the prediction function and display the results.
Add extra features: extra features such as flower language interpretation.
Here I referred to multiple online tutorials and ended up with my PyQt5 code thus code: [https://github.com/qq1308636759/VGG16--/blob/main/UI.py](https://github.com/qq1308636759/VGG16--/blob/main/UI.py).

The third-party code snippets used for reference are as follows:
```
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(765, 402)
        self.centralwidget = QtWidgets.QWidget(Form)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(70, 50, 256, 256))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(560, 300, 151, 61))
        self.pushButton.setObjectName("pushButton")
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(420, 50, 256, 51))
        self.textBrowser.setStyleSheet("border:0px;\n""")
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 300, 151, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser_1 = QtWidgets.QTextBrowser(Form)
        self.textBrowser_1.setGeometry(QtCore.QRect(420, 140, 261, 101))
        self.textBrowser_1.setObjectName("textBrowser1")
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.pushButton.clicked.connect(self.prediction)
        self.pushButton_2.clicked.connect(self.openimg)
```
(see “ui.py” for the exact code)

## 5. Testing and optimisation
### User interface testing: 
Ensure that the application can run stably under different systems and environments, and that the interface is friendly and intuitive.

### Documentation of the commissioning process:
The first is the learning rate lr = 0.01,0.005,0.001,0.0005 which is finally determined to be 0.001
During the debugging process, the learning rate and optimiser were adjusted and compared several times. Firstly the learning rate lr = 0.01,0.005,0.001,0.0005 It was finally determined that the learning rate of 0.001 gave the best results.
For the choice of optimiser, both SGD and Adam were tried and finally the Adam optimiser was decided to be used. In addition, the decision to use a lightweight model was confirmed at an early stage, which helped to increase the training speed and increased the training efficiency.

Ultimately, the accuracy of the model on the validation set was about 91.96%.

### Operation Log Screenshot:
<img width="640" alt="image" src="https://github.com/Quincy816/Coding3Final2.0/assets/115622644/291809a9-e2f2-4fb1-957b-20d50cb13e65">

The line graphs for train_acc, valid_acc, train_loss, valid_loss are generated as follows:
![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/c2a0fd82-7108-4b59-8ec6-350c338a9f58)
![image](https://github.com/Quincy816/Coding3Final2.0/assets/115622644/fb72e2f3-ca04-4537-96a3-2861c66b7ec7)

## 6. Conclusion
Overall, I evaluate the dataset in three ways, the first is the reference of the output accuracy values in the terminal, the second is the actual practice where I can accurately identify the type of flowers with any random picture. Secondly the output of the line graphs of train_acc, valid_acc, train_loss, valid_loss helps me to find the most suitable parameters when adjusting the hyperparameters.
