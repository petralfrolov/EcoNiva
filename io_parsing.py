import streamlit as st
import pandas as pd
import pdfplumber
from io import BytesIO
from config import FEED_MAP

# --- ВСПОМОГАТЕЛЬНОЕ: общая нормализация таблицы и агрегация ---
def _aggregate_table(df_raw, logs):
    # Ожидаем хотя бы 2 колонки, ищем нужные по подстрокам
    logs.append(f"Обнаружены колонки: {list(df_raw.columns)}")
    ingredient_col = next((c for c in df_raw.columns if 'Ингредиент' in str(c)), None)
    sv_kg_col      = next((c for c in df_raw.columns if 'СВ кг' in str(c)), None)
    if not ingredient_col or not sv_kg_col:
        logs.append("ОШИБКА: В таблице не найдены обязательные колонки 'Ингредиенты' или 'СВ кг'.")
        return {}, logs

    logs.append(f"Колонка ингредиентов: '{ingredient_col}', колонка данных: '{sv_kg_col}'")
    df = df_raw[[ingredient_col, sv_kg_col]].copy()

    # Числа: запятая->точка, coercion
    df[sv_kg_col] = pd.to_numeric(df[sv_kg_col].astype(str).str.replace(',', '.'), errors='coerce')
    df.dropna(inplace=True)
    logs.append(f"Найдено {len(df)} строк с числовыми данными.")

    # Очистка имени ингредиента
    df['clean_name'] = (df[ingredient_col].astype(str).str.split('/', n=1).str[0]
                        .str.replace(r'\s{2,}', ' ', regex=True)
                        .str.strip('.,\\ '))

    aggregated_data = {key: 0.0 for key in FEED_MAP.keys()}

    # Лог первых строк
    logs.append("\n--- Построчное сопоставление (первые 15 строк) ---")
    for index, row in df.head(15).iterrows():
        match_found = False
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += float(row[sv_kg_col])
                logs.append(
                    f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✓ '{category}' ({row[sv_kg_col]:.2f} кг)")
                match_found = True
                break
        if not match_found:
            logs.append(
                f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✗ Категория не найдена")

    # Остальные строки без логов построчно
    for _, row in df.iloc[15:].iterrows():
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += float(row[sv_kg_col])
                break

    logs.append("\n--- Итог агрегации ---")
    logs.append(str({k: round(v, 2) for k, v in aggregated_data.items() if v > 0}))

    if sum(aggregated_data.values()) < 0.01:
        logs.append("ПРЕДУПРЕЖДЕНИЕ: Сумма всех найденных компонентов равна нулю. Данные не загружены.")
        return {}, logs

    return aggregated_data, logs


@st.cache_data
def parse_pdf_report(uploaded_file):
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."]

    logs.append(f"Начало обработки файла: {uploaded_file.name}")
    tables = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for _, page in enumerate(pdf.pages):
                extracted = page.extract_tables()
                if extracted:
                    tables.extend(extracted)
        logs.append(f"Найдено таблиц в PDF: {len(tables)}")
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении PDF: {e}")
        return {}, logs

    if not tables or len(tables[0]) < 2:
        logs.append("ОШИБКА: Не удалось найти подходящую таблицу с данными в PDF.")
        return {}, logs

    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
    # В некоторых PDF первая строка — дубль заголовков
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    return _aggregate_table(df, logs)


# --- НОВОЕ: Excel-парсер ---
@st.cache_data
def parse_excel_report(uploaded_file):
    logs = []
    if uploaded_file is None:
        return {}, ["Файл не был загружен."]

    logs.append(f"Начало обработки Excel: {uploaded_file.name}")
    try:
        # Читаем первую подходящую вкладку: где есть 'Ингредиент' и 'СВ кг'
        xls = pd.ExcelFile(uploaded_file)
        candidate = None
        for sheet in xls.sheet_names:
            df_try = xls.parse(sheet)
            cols = [str(c) for c in df_try.columns]
            if any('Ингредиент' in c for c in cols) and any('СВ кг' in c for c in cols):
                candidate = df_try
                logs.append(f"Выбрана вкладка: {sheet}")
                break
        if candidate is None:
            # если не нашли — берём первый лист
            logs.append("Подходящих вкладок не найдено, используем первый лист.")
            candidate = xls.parse(xls.sheet_names[0])
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении Excel: {e}")
        return {}, logs

    return _aggregate_table(candidate, logs)


# --- НОВОЕ: Универсальный диспатчер ---
@st.cache_data
def parse_any_report(uploaded_file):
    if uploaded_file is None:
        return {}, ["Файл не был загружен."]
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return parse_pdf_report(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return parse_excel_report(uploaded_file)
    return {}, [f"Неподдерживаемый формат: {uploaded_file.name}. Ожидается PDF или Excel."]
