from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2Tokenizer, LayoutLMv2Processor, LayoutLMv2ForSequenceClassification
import torch
from tqdm.notebook import tqdm
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets import Dataset
import pandas as pd
import numpy as np
import pytesseract
import img2pdf
import requests
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def classify_img():
    image = Image.open(
        "./Data Base/PAN Card/PAN-Card.tiff")
    image = image.convert("RGB") or image.convert("RGBA")
    print("Image Function")
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])

    feature_extractor = LayoutLMv2FeatureExtractor()
    tokenizer = LayoutLMv2Tokenizer.from_pretrained(
        "microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)

    encoded_inputs = processor(image, return_tensors="pt")

    processor.tokenizer.decode(encoded_inputs.input_ids.squeeze().tolist())

    dataset_path = "./Data Base"
    labels = [label for label in os.listdir(dataset_path)]
    # labels.flags.allows_duplicate_labels = False
    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}

    images = []
    labels = []

    for label_folder, _, file_name in os.walk(dataset_path):
        if label_folder != dataset_path:
            label = label_folder[12:]
            for _, _, image_names in os.walk(label_folder):
                relative_image_names = []
                for image_file in image_names:
                    relative_image_names.append(
                        dataset_path + "/" + label + "/"+image_file)
                images.extend(relative_image_names)
                labels.extend([label] * len(relative_image_names))
                labels1 = list(dict.fromkeys(labels))

    data = pd.DataFrame.from_dict({"image_path:": images, "label": labels})
    dataset = Dataset.from_pandas(data)

    features = Features({
        'image': Array3D(dtype="int64", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': ClassLabel(num_classes=len(labels1), names=labels1),
    })

    def preprocess_data(datasets):
        images = [Image.open(path).convert("RGB")
                  for path in datasets["image_path:"]]
        encoded_inputs = processor(
            images, padding="max_length", truncation=True)

        encoded_inputs["labels"] = [label2id[label]
                                    for label in datasets["label"]]
        return encoded_inputs

    encoded_datasets = dataset.map(
        preprocess_data, remove_columns=dataset.column_names, features=features, batched=True, batch_size=2)

    encoded_datasets.set_format(type="torch", device="cpu")
    dataloader = torch.utils.data.DataLoader(encoded_datasets, batch_size=4)
    batch = next(iter(dataloader))

    processor.tokenizer.decode(batch["input_ids"][0].tolist())

    test = id2label[batch['labels'][0].item()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",
                                                                num_labels=len(labels))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    global_step = 0
    num_train_epochs = 1
    # total number of training steps
    t_total = len(dataloader) * num_train_epochs

    # put the model in training mode
    model.train()
    for epoch in range(num_train_epochs):
        print("Epoch:", epoch)
        running_loss = 0.0
        correct = 0
        for batch in tqdm(dataloader):
            # forward pass
            outputs = model(**batch)
            loss = outputs.loss

            running_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == batch['labels']).float().sum()

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / len(data)
        print("Training accuracy:", accuracy.item())

    file = open("testdocument2.pdf", "wb")
    file.write(img2pdf.convert("testdocument.png"))
    file.close()

    poppler_path = r"E:\MCA STUDY\MACHINE LEARNING\poppler-23.01.0\Library\bin"
    pdf_path = r"G:\icosmic_submission\testdocument2.pdf"
    pages = convert_from_path(pdf_path=pdf_path, poppler_path=poppler_path)
    save_folder = r"G:\icosmic_submission"
    save_folder2 = r"G:\icosmic_submission\Images"
    c = 1

    for page in pages:
        img_name = f"img-{c}.png"
        page.save(os.path.join(save_folder, img_name), "PNG")
        img_name = Image.open(img_name)
        img_name = img_name.convert("RGB")

        encoded_inputs = processor(
            img_name, return_tensors="pt", truncation=True)
        for k, v in encoded_inputs.items():
            encoded_inputs[k] = v.to(model.device)

        outputs = model(**encoded_inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Page {c} Classification:")
        img_name_class = id2label[predicted_class_idx]
        print(img_name_class)
        img_name2 = img_name_class + f"{c}" + ".png"
        page.save(os.path.join(save_folder2, img_name2), "PNG")
        c += 1
