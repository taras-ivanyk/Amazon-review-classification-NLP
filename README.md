# Amazon 5-Star Review Classification Pipeline

Проєкт NLP для багатокласової класифікації (1–5 зірок) відгуків Amazon.

## Структура

- `data/` — датасети train/test
- `main.py` — весь pipeline (очищення тексту → Bag-of-Words → TF-IDF → Naive Bayes)
- `requirements.txt` — залежності
- `.gitignore` — ігноровані файли

## Запуск

```bash
pip install -r requirements.txt
python main.py
