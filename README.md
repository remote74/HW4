# Sentiment Analysis с BERT

## Описание
Проект по анализу тональности текстов с использованием трансформеров.

## Структура
- `fine_tuned_model/` — обученная модель
- `src/app.py` — Gradio приложение
- `src/model_repository.py` — инициализация модели и токенайзера
- `artifacts/error_analysis.txt` — анализ ошибок
- `artifacts/comparison_results.txt` — сравнение моделей

## Запуск демо

### Вариант с Gradio:
```bash
pip install gradio transformers torch
python app.py
```
Откройте http://127.0.0.1:7860

## Результаты

### Fine-tuned модель:
- F1 (macro): 0.8876
- Accuracy: 0.8877

### Baseline модель:
- F1 (macro): 0.8211
- Accuracy: 0.8211

Улучшение: 8.11%

## Использование в коде
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model_ft = AutoModelForSequenceClassification.from_pretrained(model_path)

# Предсказание
inputs = tokenizer("Your text here", return_tensors="pt")
outputs = model(**inputs)
pred = torch.argmax(outputs.logits, dim=1)
```

## Требования
- Python 3.8+
- transformers
- torch
- gradio (для демо)