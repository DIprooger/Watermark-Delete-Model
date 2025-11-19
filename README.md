
<details>
<summary>🇬🇧 English</summary>

# Watermark-Delete-Model

A neural network project for removing watermarks from images.

## 🧠 Description

This repository includes:

- `infer_one.py`: batch image cleaning using a trained Unet++ model.
- `dataset.py`: watermark augmentation using logos, with support for blend modes and random placement.
- `watermark-segmentation.ipynb`: notebook for training and testing.
- `my_image/`, `my_logo/`: folders for clean images and watermark logos.

## 🚀 How to use

1. Place input images into the `image_for_cleare/` folder.
2. Make sure `watermark_model.pth` is present in the project root.
3. Run:
   ```bash
   python infer_one.py
   ```
4. Cleaned images will be saved to the `output_cleaned/` folder.

## 🧪 Training

The model is trained using synthetic data generated from clean images and watermark logos.  
The `Dataset` class:

- overlays logos with different **blend modes** (`normal`, `multiply`, `overlay`),
- randomizes **position, scale, rotation**,
- creates a **binary mask** of watermark regions.

## 📚 How to Train the Watermark Removal Model

This guide explains how to train a neural network to remove watermarks using synthetic data generation.

---

### 🧠 Model

The model is based on:

- **Architecture**: Unet++
- **Encoder**: ResNet34
- **Library**: `segmentation_models_pytorch`

---

### 📁 Folder Structure

```bash
.
├── my_image/                   # Your clean images
├── my_logo/                    # Watermark logos (PNG with transparency)
├── dataset.py                  # Synthetic watermark generator
├── watermark-segmentation.ipynb  # Training notebook
├── requirements.txt
```

---

### ⚙️ Step-by-Step Training

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare training data**

   - Put thousands of **clean images** (without watermarks) into the `my_image/` folder.  
     The more and the closer they are to your real use case (same site / camera / content), the better the model will generalize.
   - Put your **watermark logos** into the `my_logo/` folder:
     - preferably **transparent PNG** without background,
     - shape, colors and opacity should **match real watermarks** you want to remove,
     - 5–10 different logos are usually enough if they are similar to real ones.

   > 💡 If you only have one “real” watermark (e.g. from some website), you can try to **extract it from a sample image in Photoshop** (remove background, export as PNG) and add it into `my_logo/`.

3. **Open and run the notebook**

   ```bash
   jupyter notebook watermark-segmentation.ipynb
   ```

   In the notebook:

   - `Dataset` loads images from `my_image/` and logos from `my_logo/`.
   - Synthetic watermarks are generated **on the fly** with random:
     - positions,
     - scale,
     - rotation,
     - blend modes and opacity.
   - The model is trained as a **segmentation network** to predict the watermark mask.

4. **Save trained weights**

   After training, export the model:

   ```python
   torch.save({"model." + k: v for k, v in model.state_dict().items()}, "watermark_model.pth")
   ```

   This file can be used for inference in `infer_one.py`.

---

### 🔁 Dataset Generation Logic

For each training image, the dataset:

- Applies **1 to 5 logos** per image.
- Uses random **position, scale and rotation**.
- Chooses between blend modes (`normal`, `multiply`, `overlay`).
- Generates a **binary mask** where watermark pixels are marked as 1.

See `dataset.py` for implementation details.

---

### ✅ Practical Training Tips

- **Lots of data helps**
  - Aim for **thousands of clean images**.
  - If you plan to clean photos from a specific site, try to use **similar photos** for training (same style, resolution, subjects).

- **Logos matter**
  - Use the **same or very similar watermarks** as in your real data.
  - If real watermarks are **opaque (no transparency)**, you can also generate them without transparency in the dataset — the model will better learn that exact style.
  - But: if you train only on dense opaque logos, there is a higher risk that the model will start **removing real text from the image** (titles, labels, etc.).

- **Training length**
  - Increase the number of **training epochs / steps** until the **validation IoU / Dice** stabilizes.
  - As a rough reference, one successful training run took about **2 days** with ~**400 steps per epoch** (your numbers will depend on GPU and batch size).

- **Ask AI for help**
  - You can use an assistant (like this one) to:
    - pick learning rate, batch size, number of epochs,
    - debug training curves,
    - adjust augmentations and loss functions.

If the model fails to remove a particular watermark, it is very likely that **this watermark (or one very similar to it)** was **not present in training**. In that case, extract this logo (e.g. via Photoshop) or find it at the source, add it to `my_logo/` and retrain or fine-tune the model.

</details>


<details>
<summary>🇷🇺 Русский</summary>

# Watermark-Delete-Model

Нейросетевой проект для удаления водяных знаков с изображений.

## 🧠 Описание

Репозиторий включает:

- `infer_one.py`: пакетная очистка изображений с помощью обученной модели Unet++.
- `dataset.py`: генерация синтетических вотермарок с логотипами, поддержка режимов наложения и случайного размещения.
- `watermark-segmentation.ipynb`: ноутбук для обучения и тестирования.
- `my_image/`, `my_logo/`: папки с исходными изображениями и логотипами.

## 🚀 Как использовать

1. Поместите изображения в папку `image_for_cleare/`.
2. Убедитесь, что файл `watermark_model.pth` находится в корне проекта.
3. Запустите:
   ```bash
   python infer_one.py
   ```
4. Очищенные изображения появятся в папке `output_cleaned/`.

## 🧪 Обучение

Модель обучается на синтетических данных, где к чистым изображениям добавляются логотипы с разной прозрачностью и режимами наложения.  
Класс `Dataset`:

- накладывает логотипы в режимах `normal`, `multiply`, `overlay`,
- рандомизирует позицию, масштаб и поворот,
- создаёт бинарную маску областей с водяными знаками.

## 📚 Как обучить модель удаления водяных знаков

Это руководство объясняет, как обучить нейросетевую модель удалению водяных знаков с помощью синтетически сгенерированных данных.

---

### 🧠 Модель

Модель основана на:

- **архитектуре**: Unet++
- **энкодере**: ResNet34
- **библиотеке**: `segmentation_models_pytorch`

---

### 📁 Структура проекта

```bash
.
├── my_image/                   # Исходные (чистые) изображения
├── my_logo/                    # Логотипы водяных знаков (PNG с прозрачностью)
├── dataset.py                  # Генератор синтетических водяных знаков
├── watermark-segmentation.ipynb  # Ноутбук для обучения
├── requirements.txt
```

---

### ⚙️ Пошаговое обучение

1. **Установка зависимостей**

   ```bash
   pip install -r requirements.txt
   ```

2. **Подготовка данных**

   - Поместите **тысячи чистых фотографий** (без водяных знаков) в папку `my_image/`.  
     Чем больше и чем сильнее они похожи на реальные (с которых вы потом будете удалять вотермарки — тот же сайт/камера/тематика), тем лучше обучится модель.
   - Поместите логотипы водяных знаков в папку `my_logo/`:
     - форматы: **PNG с прозрачностью**, без фона,
     - форма, цвета и прозрачность должны **максимально совпадать** с реальными вотермарками,
     - 5–10 разных логотипов обычно достаточно, если они похожи на реальные.

   > 💡 Если у вас есть только одна реальная вотермарка (например, с конкретного сайта), её можно **аккуратно “добыть” из фото в Photoshop**: вырезать без фона, сохранить как PNG и добавить в `my_logo/`.

3. **Откройте и запустите ноутбук**

   ```bash
   jupyter notebook watermark-segmentation.ipynb
   ```

   В ноутбуке:

   - класс `Dataset` читает изображения из `my_image/` и логотипы из `my_logo/`;
   - во время обучения “на лету” генерируются:
     - от 1 до 5 логотипов на изображение,
     - случайные позиция, масштаб, поворот,
     - разные режимы наложения и прозрачность;
   - модель обучается как **сегментационная**: предсказывает маску водяных знаков.

4. **Сохраните обученную модель**

   После обучения выполните:

   ```python
   torch.save({"model." + k: v for k, v in model.state_dict().items()}, "watermark_model.pth")
   ```

   Файл `watermark_model.pth` используется в `infer_one.py` для удаления водяных знаков.

---

### 🔁 Генерация синтетического датасета

Генерация данных происходит "на лету":

- от 1 до 5 логотипов на одно изображение;
- случайные позиция, масштаб, поворот;
- режимы наложения: `normal`, `multiply`, `overlay`;
- создаётся бинарная маска областей водяных знаков.

Подробнее см. в `dataset.py`.

---

### ✅ Практические советы по обучению

- **Много исходных фотографий**
  - Старайтесь использовать **тысячи чистых фото**.
  - Лучше, если они максимально **похожи на те**, с которых вы будете удалять водяные знаки (тот же сайт, ракурс, качество).

- **Правильные логотипы**
  - Важно, чтобы модель видела **те же самые или очень похожие вотермарки**, что и в бою.
  - Если в реальных данных логотип **без прозрачности**, можно генерировать такие же плотные логотипы и в датасете — модель лучше подстроится.
  - Но помните: если обучать только на “жирных” логотипах без прозрачности, повышается риск, что модель начнёт **удалять полезный текст** на фото (подписи, номера и т.п.).

- **Сколько учить**
  - Увеличивайте количество **эпох / шагов обучения**, пока метрики (IoU, Dice) на валидации **перестанут расти** и стабилизируются.
  - В одном из успешных запусков обучение занимало порядка **двух дней**, при этом было около **400 шагов на эпоху** (конкретные цифры зависят от вашей видеокарты, батча и объёма данных).

- **Можно спрашивать ИИ**
  - Помощник (как этот) может помочь:
    - подобрать learning rate, batch size, количество эпох,
    - разобраться с переобучением/недообучением,
    - настроить аугментации и функции потерь.

Если на каких-то картинках водяной знак не удаляется, почти всегда причина в том, что **модель ни разу не видела такую вотермарку** во время обучения. В этом случае попробуйте:

1. Вытащить логотип из реального фото (через Photoshop или аналог),
2. Добавить его в `my_logo/`,
3. Дообучить (fine-tune) или переобучить модель.

</details>
