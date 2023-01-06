<h1 align="center">Serapis AI Image Classifier</h1>

<p align="center">
  <img src="https://user-images.githubusercontent.com/73674035/211046713-2a96c5f3-6842-48d1-852c-57db0e007455.png" alt="The Staff of Serapis"/>
</p>

<p align="center">
  Serapis AI Image Classifier is a program that allows you to automatically create image datasets using <a href="https://serpapi.com/images-results">SerpApi's Google Images Scraper API</a>, finetune a ResNet50 model, and classify images using the trained model.
<br>
</p>


---

<h3 align="center">Installation</h3>

You can install these dependencies using the following command:
```bash
pip install -r requirements.txt
```

---

<h3 align="center">Usage</h3>

You can use Serapis AI Image Classifier in one of the following three modes:

---

<h3 align="center">Create a dataset and train a new model from scratch</h3>

To create a dataset and train a new model from scratch, you will need to provide a list of labels and an image to use as a reference for the scraping process.
[SerpApi API Key](https://serpapi.com/manage-api-key) is necessary for this mode for the program to automatically scrape images you will use in your database using [SerpApi's Google Images Scraper API](https://serpapi.com/images-results).

You can [register to SerpApi to claim free credits](https://serpapi.com/users/sign_up).

```bash
python serapis.py --train --labels eagle, bull, lion, man --image-path lionimage.jpg --api-key <SerpApi-API-KEY>
```

---

<h3 align="center">Use old scraped images and train a new model</h3>

To use old scraped images and train a new model, you will need to provide a list of labels and specify that you want to use old images with the --use-old-images flag.
You can also put your images in the `images/` folder and also add enter them in `images/catalogue.csv` to manually train models using your own dataset.

```bash
python serapis.py --train --labels eagle, bull, lion, man --use-old-images --image-path lionimage.jpg
```

---

<h3 align="center">Use a previously trained model</h3>

To use a previously trained model, you will need to provide the path to the trained model and an image to classify.

```bash
python serapis.py --model-path models/1.pth --image-path lionimage.jpg
```

---

<h3 align="center">Dialogue Mode</h3>

You can also navigate the program by not providing any arguments and using the dialogue mode:

```bash
python serapis.py
```

---

<h3 align="center">The Output</h3>
<p align="center">
<img src="https://user-images.githubusercontent.com/73674035/211052320-f53cc530-6047-4ac3-8177-53f3daa6371a.png" alt="Classified Image of a Lion"/>
</p>

The output will give you the answer:

```bash
The image contains Lion
```

---

Optional Arguments:

    -h, --help                    Help to Nagigate
    --train                       Whether to train a new model
    --model-path MODEL_PATH       Pretrained Model path you want to use
    --dialogue                    Whether to use dialogue to navigate through the program
    --use-old-images              Whether to use old images you have downloaded to train a new model
    --api-key API_KEY             SerpApi API Key
    --limit LIMIT                 Number of images you want to scrape at most for each label
    --labels LABELS [LABELS ...]  Labels you want to use to train a new model
    --image-path IMAGE_PATH       Path to the image you want to classify
