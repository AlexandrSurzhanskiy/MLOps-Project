# Прогнозирование интереса клиента к продукту

## Цель проекта

Цель проекта — построить модель машинного обучения, которая по историческим взаимодействиям клиентов с продуктами способна предсказать вероятность того, что конкретный клиент проявит интерес к новому продукту.  

Модель может использоваться для:
- персонализации продуктовых предложений;
- анализа предпочтений клиентов и прогнозирования спроса;
- поддержки продуктовых решений и планирования ассортимента.

---

## Целевые метрики

| Категория | Метрика | Целевое значение | Цель |
|------------|----------|------------------|------|
| **Качество модели** | ROC AUC | ≥ 0.80 | Разделять заинтересованных и незаинтересованных клиентов |
| | F1-score | ≥ 0.05 | Баланс точности и полноты |
| | Accuracy | ≥ 0.90 | Общая корректность классификации |
| **Бизнес-метрики** | Uplift ROI | ≥ +3% | Прирост возврата инвестиций по сравнению с базовой рассылкой |
| **Сервисные метрики** | Среднее время инференса | ≤ 200 мс | Возможность онлайн-применения |
| | Доля ошибок при инференсе | ≤ 1 % | Стабильность в эксплуатации |

---

## Описание данных

Данные представляют собой взаимодействия пользователей с продуктами или услугами:

| user_idx | item_idx | label |
|-----------|-----------|-------|
| 0         | 101       | 1     |
| 1         | 105       | 0     |
| 2         | 110       | 1     |

**Описание полей:**
- `user_idx` — идентификатор пользователя;  
- `item_idx` — идентификатор продукта или услуги;  
- `label` — бинарный индикатор интереса (`1` — проявил интерес, `0` — нет).  

Формат хранения: **Parquet** (`data/processed/test.parquet`)

Данные подготовлены для совместимости с моделями на основе нейронных сетей (эмбеддинги пользователей и продуктов).

---

## Где лежат данные и модели (DVC)

Большие файлы (данные и артефакты модели) версионируются через **DVC** и не хранятся в Git.

- **Сырые данные (DVC):** `data/raw/events.csv`
- **После предобработки (stage `prepare`):**
  - `data/processed/interactions.parquet`
  - `data/processed/train.parquet`
  - `data/processed/test.parquet`
- **Итоговая модель (stage `train`):** `models/recsys_nn_v1/`
  - веса: `models/recsys_nn_v1/pytorch_model.bin`
  - конфиг: `models/recsys_nn_v1/config.json`
  - метрики: `models/recsys_nn_v1/metrics.json`
- **Результаты (stage `evaluate`):**
  - предсказания: `predictions/recsys_nn_v1/*.csv`
  - отчёты: `reports/recsys_nn_v1/*`

---

## Как запустить проект

### 1️⃣ Установка зависимостей
```bash
pip install -r requirements.txt
````

### 2️⃣ Подготовка данных
Положите исходный файл с сырыми событиями в папку:

```bash
data/raw/events.csv
```

Запустите предобработку:
```bash
PYTHONPATH=. python pipelines/run_preprocess.py \
  --input data/raw/events.csv \
  --output_dir data/processed/
```
После выполнения появятся файлы:
```bash
data/processed/train.parquet
data/processed/test.parquet
```

### 3️⃣ Обучение модели

```bash
PYTHONPATH=. python pipelines/run_training.py \
  --config configs/train_config.yaml
```

### 4️⃣ Запуск инференса

```bash
PYTHONPATH=. python pipelines/run_inference.py \
  --model_dir models/recsys_nn_v1/ \
  --data data/processed/test.parquet
```

### 5️⃣ Оценка качества модели

```bash
PRED=$(ls -t predictions/recsys_nn_v1/predictions_*.csv | head -n 1)

PYTHONPATH=. python pipelines/run_evaluation.py \
  --predictions "$PRED" \
  --ground_truth data/processed/test.parquet \
  --output reports/recsys_nn_v1
```

---

## Как восстановить и воспроизвести результат

```bash
git clone -b feature/homework2 --single-branch <REPO_URL>
cd <REPO_DIR>

pip install -r requirements.txt

dvc pull
dvc repro
```

После этого будут доступны:
  - данные в `data/processed/`
  - модель в `models/recsys_nn_v1/`
  - предсказания в `predictions/`
  - отчёты в `reports/`

---

## Трекинг экспериментов

Каждый запуск обучения создаёт отдельный MLflow run с параметрами, метриками и артефактами (веса модели, конфиг, отчёт, dvc.lock).

### Где смотреть результаты

По умолчанию используется локальное хранилище `./mlruns`.

Запуск UI:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Откройте: [http://localhost:5000](http://localhost:5000)

---

## Офлайн-инференс

### Сборка образа

Если модель/данные версионируются DVC, сначала подтяните веса модели:
```bash
dvc pull models/recsys_nn_v1
````

Соберите образ:

```bash
docker build -t ml-app:v1 .
```

### Запуск инференса

Контейнер запускает `python -m src.predict` и ожидает входной файл с колонками `user_idx`, `item_idx`.

Пример (вход Parquet → выход CSV):

```bash
docker run --rm \
  -v "$PWD/data/processed:/data" \
  ml-app:v1 \
  --input_path /data/test.parquet \
  --output_path /data/preds.csv
```

Результат будет сохранён на хосте в `data/processed/preds.csv` с добавленной колонкой `predicted_score`.

---

## План развития

* Добавить временные признаки (time-based features);
* Интегрировать данные из внешних источников (поведение, транзакции);
* Сравнить с бустинг-моделями (CatBoost, LightGBM);
* Реализовать мониторинг качества модели в продакшене.

---
