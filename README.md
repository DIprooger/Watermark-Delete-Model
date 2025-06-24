
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

The model is trained using synthetic data generated from clean images and watermark logos. The `Dataset` class applies realistic logo overlays with different blend modes and opacities.

## 🧰 Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```


# 📚 How to Train the Watermark Removal Model

This guide explains how to train a neural network to remove watermarks using synthetic data generation.

---

## 🧠 Model

The model is based on:

- **Architecture**: Unet++
- **Encoder**: ResNet34
- **Library**: `segmentation_models_pytorch`

---

## 📁 Folder Structure

```
.
├── my_image/             # Your clean images
├── my_logo/              # Watermark logos (PNG format with transparency)
├── dataset.py            # Synthetic watermark generator
├── watermark-segmentation.ipynb  # Training notebook
├── requirements.txt
```

---

## ⚙️ Step-by-Step Training

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare training data**

- Place your clean images into the `my_image/` folder.
- Place watermark logos (transparent PNGs) into the `my_logo/` folder.

3. **Open notebook**

Run and modify the notebook:

```bash
jupyter notebook watermark-segmentation.ipynb
```

- It loads the synthetic dataset using `Dataset` class.
- Applies augmentation and trains the segmentation model.

4. **Save trained weights**

After training, export the model:

```python
torch.save({"model." + k: v for k, v in model.state_dict().items()}, "watermark_model.pth")
```

This file can be used for inference in `infer_one.py`.

---

## 🔁 Dataset Generation Logic

The dataset dynamically generates synthetic watermarked images and masks:
- Applies 1 to 5 logos per image.
- Random positions, scale, rotation.

</details>


<details>
<summary>🇷🇺 Русский</summary>

# Watermark-Delete-Model

Нейросетевой проект для удаления водяных знаков с изображений.

## 🧠 Описание

Репозиторий включает:

- `infer_one.py`: пакетная очистка изображений с помощью обученной модели Unet++.
- `dataset.py`: генерация синтетических вотермарок с логотипами, поддержка режимов наложения.
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

Модель обучается на синтетических данных, где к изображениям добавляются логотипы с разной прозрачностью и режимами наложения. Класс `Dataset` отвечает за генерацию таких данных.

## 🧰 Зависимости

Установка зависимостей:

```bash
pip install -r requirements.txt
```


# 📚 Как обучить модель удаления водяных знаков

Это руководство объясняет, как обучить нейросетевую модель удалению водяных знаков с помощью синтетически сгенерированных данных.

---

## 🧠 Модель

Модель основана на:

- **Архитектуре**: Unet++
- **Энкодере**: ResNet34
- **Библиотеке**: `segmentation_models_pytorch`

---

## 📁 Структура проекта

```
.
├── my_image/             # Исходные (чистые) изображения
├── my_logo/              # Логотипы водяных знаков (в формате PNG с прозрачностью)
├── dataset.py            # Генератор синтетических водяных знаков
├── watermark-segmentation.ipynb  # Ноутбук для обучения
├── requirements.txt
```

---

## ⚙️ Пошаговое обучение

1. **Установка зависимостей**

```bash
pip install -r requirements.txt
```

2. **Подготовка данных**

- Поместите изображения без водяных знаков в папку `my_image/`.
- Поместите логотипы (с прозрачностью) в `my_logo/`.

3. **Откройте и запустите ноутбук**

```bash
jupyter notebook watermark-segmentation.ipynb
```

- Используется класс `Dataset` для генерации обучающих данных.
- Применяются аугментации и начинается обучение модели сегментации.

4. **Сохраните обученную модель**

После обучения выполните:

```python
torch.save({"model." + k: v for k, v in model.state_dict().items()}, "watermark_model.pth")
```

Файл `watermark_model.pth` можно использовать в `infer_one.py`.

---

## 🔁 Генерация синтетического датасета

Генерация данных происходит "на лету":
- От 1 до 5 логотипов на изображение.
- Случайные позиция, масштаб, поворот.
- Режимы наложения: `normal`, `multiply`, `overlay`.
- Создаётся бинарная маска зон водяных знаков.

Подробности см. в `dataset.py`.

---

## ✅ Советы

- Используйте разнообразные логотипы.
- Для захвата границ логотипов можно увеличить `dilate` в классе `Dataset`.
- Для оценки качества используйте метрики IoU или Dice.
- 
</details>
