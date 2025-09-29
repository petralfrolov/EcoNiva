import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import pickle
import os
import statsmodels.api as sm
import plotly.graph_objects as go
from pathlib import Path

# --- КОНФИГУРАЦИЯ И КОНСТАНТЫ ---

st.set_page_config(layout="wide", page_title="Анализатор рациона коров")

# Словарь для группировки сырых ингредиентов (предоставлен пользователем)
FEED_MAP = {
    'Силос': ['Силос', 'силос', 'силос_', 'C-C', 'c-c', 'С-С', 'с-с', 'С-С 2242.05.05.02.1.23\\'],
    'Сенаж': ['Сенаж', 'С-Ж', 'с-ж', 'Люцерна', 'люцерна', 'люц_', 'суданка', 'рожь_'],
    'Сено': ['Сено из люцерны', 'Сено луговое', 'Сено луговое ЛБ', 'Сено люцерна ЛБ'],
    'Солома': ['Солома', 'солома', 'Солома ЭНА Восточное -', 'солома ячменная', 'Пшеничная солома', 'Солома пшеничная',
               'Солома пшеничная ЛБ', 'Солома-пшеничная ОМ Север', 'Дробленая солома КН Восток'],
    'Кукуруза': ['Кукуруза', 'кукуруза', 'Кукуруза влажная, кр', 'Кукуруза. 70%,сухая средн.\nпомол, ЭНАПКХ',
                 'Кукуруза. 70%,сухая, мелк.\nпомол, ЭНАПКХ', 'Плющенная кукуруза ток', 'плющенная кукуруза',
                 'Плющ зерно 6203.06.05.06.1.24', 'кукуруза_плющ_', 'плющенное'],
    'Зерновые_прочие': ['Тритикале', 'Ячмень', 'Ячмень. сухой', 'Ячмень. сухой 53%, средний\nпомол, ЭНАПКХ',
                        'Ячмень.61%, сухой, средний\nпомол, ЭНАПКХ', 'Пшеница', 'пшеница_консерв_СНБ'],
    'Корнаж_ЗСК': ['Корнаж', 'корнаж', 'карнаж', 'корнаж_', 'ЗСК', 'ЗСК 6202.05.05.07.1.23', 'к-ж', 'к-ж Детч'],
    'Шрот_соевый': ['Шрот соевый'],
    'Шрот_рапсовый': ['Шрот рапсовый', 'Шрот Рапсовый'],
    'Шрот_подсолнечный': ['Шрот подсолнечный'],
    'Жмых_рапсовый': ['Жмых рапсовый', 'Жмых рапсовый СП'],
    'Жмых_льняной': ['Жмых льняной'],
    'Побочные_свекловичные': ['Жом свекловичный сухой', 'Жом свекловичный сухой. гр',
                              'Жом свекловичный сухой. гр, ЭНАПКХ v', 'Жом свекловичный сухой. гранула, ЭНАПКХ',
                              'Патока свекловичная', 'Патока свекловичная,\nсах', 'Патока свекловичная, сах',
                              'Патока свекловичная, сах.49%,\nЭНАПКХ v', 'Патока свекловичная, сах.49%, ЭНАПКХ\nv',
                              'Патока свекловичная, сахар', 'Меласса свекловичная'],
    'Побочные_прочие': ['Соевая оболочка ЭНАПКХ', 'Дробина сухая', 'Дрожжи кормовые. СП', 'Пивные дрожжи сухие'],
    'Комбикорма': ['Комбикорм', 'Комбикорм -', 'Комбикорм 10.3.3 (без пшен.)\n_Сентябрь',
                   'Комбикорм 11.3.3 (без пшен.)\n_Сентябрь', 'Комбикорм №', 'КК', 'Комбиком №', 'ККД', 'Кормосмесь',
                   'ккз аристово v'],
    'Премиксы': ['Премикс дойный А', 'Премикс дойный Б', 'Премикс дойный Б. 07.23\nЭНАПКХ', 'Премикс Транзит Б'],
    'Жиры': ['Жир защищенный', 'Жир защищеный', 'Жир защищеный. 99%,\nфракц.,ЭНАПКХ'],
    'Минеральные_добавки': ['Соль. ЭНАПКХ', 'Сода. ЭНАПКХ', 'Мел. ЭНАПКХ', 'Поташ. ЭНАПКХ', 'Кальций пропионат',
                            'Добавка ЛЕД ЭНАПКХ']
}

# Пути к файлам моделей (предоставлены пользователем)
MODEL_PATHS = {
    "Лауриновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Лауриновая.pkl'),
    "Пальмитиновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Пальмитиновая.pkl'),
    "Стеариновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Стеариновая.pkl'),
    "Олеиновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Олеиновая.pkl'),
}

# Целевые значения по ГОСТу (можно скорректировать)
TARGET_RANGES = {
    "Лауриновая": (3.0, 4.0),
    "Пальмитиновая": (30.0, 32.0),
    "Стеариновая": (8.0, 12.0),
    "Олеиновая": (20.0, 25.0),
}


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

@st.cache_data
def parse_pdf_report(uploaded_file):
    """Извлекает и агрегирует данные из PDF-отчета о рационе."""
    if uploaded_file is None:
        return {}

    tables = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_tables()
            if extracted:
                tables.extend(extracted)

    if not tables or len(tables[0]) < 2:
        st.error("Не удалось найти таблицу с данными в PDF-файле.")
        return {}

    # [cite_start]Логика очистки, основанная на предоставленном ноутбуке [cite: 228-321]
    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

    # Ищем колонки с ингредиентами и сухим веществом
    ingredient_col = next((col for col in df.columns if 'Ингредиент' in col), None)
    sv_kg_col = next((col for col in df.columns if 'СВ кг' in col), None)

    if not ingredient_col or not sv_kg_col:
        st.error("В таблице PDF не найдены колонки 'Ингредиенты' или 'СВ кг'.")
        return {}

    df = df[[ingredient_col, sv_kg_col]].copy()
    df[sv_kg_col] = pd.to_numeric(df[sv_kg_col].str.replace(',', '.'), errors='coerce')
    df = df.dropna()

    df['clean_name'] = df[ingredient_col].astype(str).str.split('/', n=1).str[0].str.strip('.,\ ')

    # Агрегация по FEED_MAP
    aggregated_data = {key: 0.0 for key in FEED_MAP.keys()}
    for _, row in df.iterrows():
        for category, names in FEED_MAP.items():
            if any(name.strip() in row['clean_name'] for name in names):
                aggregated_data[category] += row[sv_kg_col]
                break  # Переходим к следующей строке после нахождения категории

    return aggregated_data


@st.cache_data
def engineer_features(manual_inputs: dict):
    """Создает производные фичи (Sum, share) на основе ручного ввода."""
    df = pd.DataFrame([manual_inputs])
    eps = 1e-9  # Для избежания деления на ноль

    # [cite_start]Расчет суммарных объемов блоков [cite: 1099-1107]
    df['Sum_conc'] = df[['Кукуруза', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК']].sum(axis=1)
    df['Sum_rough'] = df[['Сенаж', 'Сено', 'Солома']].sum(axis=1)
    df['Sum_prot_ex'] = df[['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный']].sum(axis=1)
    df['Sum_prot_pr'] = df[['Жмых_рапсовый', 'Жмых_льняной']].sum(axis=1)
    df['Sum_prot'] = df['Sum_prot_ex'] + df['Sum_prot_pr']

    # [cite_start]Расчет долей внутри блоков [cite: 1108-1129]
    df['share_kukur'] = np.where(df['Sum_conc'] > 0, df['Кукуруза'] / (df['Sum_conc'] + eps), 0.0)
    df['share_korn'] = np.where(df['Sum_conc'] > 0, df['Корнаж_ЗСК'] / (df['Sum_conc'] + eps), 0.0)
    df['share_zern'] = np.where(df['Sum_conc'] > 0, df['Зерновые_прочие'] / (df['Sum_conc'] + eps), 0.0)
    df['share_soloma'] = np.where(df['Sum_rough'] > 0, df['Солома'] / (df['Sum_rough'] + eps), 0.0)
    df['share_prot_soy'] = np.where(df['Sum_prot'] > 0, df['Шрот_соевый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_raps'] = np.where(df['Sum_prot'] > 0, df['Шрот_рапсовый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_lin'] = np.where(df['Sum_prot'] > 0, df['Жмых_льняной'] / (df['Sum_prot'] + eps), 0.0)

    return df


@st.cache_resource
def load_models(paths: dict):
    """Загружает модели и их метаданные из pkl файлов."""
    models = {}
    for acid_name, path in paths.items():
        if not path.exists():
            st.error(f"Файл модели не найден по пути: {path}")
            return None
        with open(path, "rb") as f:
            models[acid_name] = pickle.load(f)
    return models


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---

# Инициализация состояния
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'manual_inputs' not in st.session_state:
    st.session_state.manual_inputs = {key: 0.0 for key in FEED_MAP.keys()}

# Загрузка моделей
models = load_models(MODEL_PATHS)
if models is None:
    st.stop()

# --- БОКОВАЯ ПАНЕЛЬ (САЙДБАР) ---
with st.sidebar:
    st.header("⚙️ Параметры рациона")

    uploaded_file = st.file_uploader(
        "Загрузите PDF-отчет",
        type="pdf",
        help="Данные из отчета автоматически заполнят поля ниже"
    )

    if uploaded_file:
        parsed_data = parse_pdf_report(uploaded_file)
        if parsed_data:
            st.session_state.manual_inputs = parsed_data
            st.session_state.run_analysis_on_load = True  # Флаг для авто-запуска

    st.subheader("Состав рациона (кг Сухого Вещества):")

    # Поля для ручного ввода
    for key in FEED_MAP.keys():
        st.session_state.manual_inputs[key] = st.number_input(
            key,
            min_value=0.0,
            value=st.session_state.manual_inputs[key],
            step=0.1,
            format="%.2f"
        )

    col1, col2 = st.columns(2)
    with col1:
        calculate_button = st.button("📈 Рассчитать", use_container_width=True)
    with col2:
        reset_button = st.button("Reset", use_container_width=True)

# --- ЛОГИКА ОБРАБОТКИ ---
if reset_button:
    st.session_state.manual_inputs = {key: 0.0 for key in FEED_MAP.keys()}
    st.session_state.analysis_run = False
    st.experimental_rerun()

if calculate_button or st.session_state.get('run_analysis_on_load', False):
    st.session_state.run_analysis_on_load = False  # Сбрасываем флаг

    # 1. Создаем производные фичи
    features_df = engineer_features(st.session_state.manual_inputs)

    # 2. Делаем прогнозы для каждой кислоты
    predictions = {}
    any_deviations = False
    for acid_name, model_data in models.items():
        model = model_data['model']
        model_features = model_data['features']

        # Убедимся, что все нужные фичи есть
        if not all(f in features_df.columns for f in model_features):
            st.error(f"Ошибка: для модели '{acid_name}' не хватает вычисленных признаков.")
            continue

        X = features_df[model_features]
        if model_data.get('add_constant', True):
            X = sm.add_constant(X, has_constant='add')

        # Получаем прогноз и доверительный интервал
        pred_summary = model.get_prediction(X).summary_frame(alpha=0.05)

        mean_pred = pred_summary['mean'].iloc[0]
        ci_lower = pred_summary['mean_ci_lower'].iloc[0]
        ci_upper = pred_summary['mean_ci_upper'].iloc[0]

        target_min, target_max = TARGET_RANGES[acid_name]

        if mean_pred < target_min:
            status = "🔴 Ниже нормы"
            any_deviations = True
        elif mean_pred > target_max:
            status = "🔴 Выше нормы"
            any_deviations = True
        else:
            status = "🟢 Норма"

        predictions[acid_name] = {
            "mean": mean_pred,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "target": f"{target_min:.1f}% – {target_max:.1f}%",
            "status": status,
        }

    st.session_state.predictions = predictions
    st.session_state.any_deviations = any_deviations
    st.session_state.analysis_run = True

# --- ОСНОВНАЯ ОБЛАСТЬ: ВЫВОД РЕЗУЛЬТАТОВ ---
st.title("🐄 Аналитический дашборд")
st.markdown("---")

if not st.session_state.analysis_run:
    st.info("Введите данные в панели слева или загрузите PDF-отчет для начала анализа.")
    # Отображение "серых" плейсхолдеров
    st.subheader("Прогноз по жирным кислотам")
    cols = st.columns(len(TARGET_RANGES))
    for i, acid_name in enumerate(TARGET_RANGES.keys()):
        cols[i].metric(label=acid_name, value="— %")

    st.subheader("Визуализация")
    fig = go.Figure()
    fig.update_layout(
        title="Прогноз и целевые диапазоны",
        xaxis_title="Жирная кислота",
        yaxis_title="Содержание, %",
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text="Данные для анализа отсутствуют", xref="paper", yref="paper", showarrow=False,
                          font=dict(size=20, color="grey"))]
    )
    st.plotly_chart(fig, use_container_width=True)

else:  # Если анализ был запущен
    preds = st.session_state.predictions

    st.subheader("Прогноз по жирным кислотам")
    cols = st.columns(len(preds))
    for i, (acid_name, data) in enumerate(preds.items()):
        cols[i].metric(
            label=f"{acid_name} ({data['status']})",
            value=f"{data['mean']:.2f}%",
            help=f"Целевой диапазон: {data['target']}. 95% доверительный интервал: {data['ci_lower']:.2f}% – {data['ci_upper']:.2f}%"
        )

    st.subheader("Детальная таблица")
    df_results = pd.DataFrame(preds).T  # Транспонируем для удобства
    df_results.rename(columns={'mean': 'Прогноз', 'target': 'Целевой диапазон', 'status': 'Статус'}, inplace=True)
    df_results['Дов. интервал (95%)'] = df_results.apply(lambda row: f"{row['ci_lower']:.2f} – {row['ci_upper']:.2f}",
                                                         axis=1)
    st.dataframe(df_results[['Прогноз', 'Дов. интервал (95%)', 'Целевой диапазон', 'Статус']], use_container_width=True)

    if st.session_state.any_deviations:
        st.subheader("⚠️ Рекомендации по корректировке")
        # Тут можно добавить более сложную логику, основанную на коэффициентах моделей
        st.warning("Обнаружены отклонения от нормы. Рассмотрите возможность изменения состава рациона.")

    st.subheader("Визуализация")
    fig = go.Figure()

    acid_names = list(preds.keys())
    mean_values = [p['mean'] for p in preds.values()]
    ci_lower = [p['ci_lower'] for p in preds.values()]
    ci_upper = [p['ci_upper'] for p in preds.values()]

    # Добавляем бары с прогнозами
    fig.add_trace(go.Bar(
        x=acid_names,
        y=mean_values,
        error_y=dict(
            type='data',
            symmetric=False,
            array=np.array(ci_upper) - np.array(mean_values),
            arrayminus=np.array(mean_values) - np.array(ci_lower)
        ),
        name='Прогнозное значение'
    ))

    # Добавляем целевые зоны
    for i, acid_name in enumerate(acid_names):
        min_target, max_target = TARGET_RANGES[acid_name]
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=i - 0.4, y0=min_target,
            x1=i + 0.4, y1=max_target,
            fillcolor="lightgreen",
            opacity=0.5,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title="Прогноз и целевые диапазоны",
        xaxis_title="Жирная кислота",
        yaxis_title="Содержание, %",
        legend_title="Обозначения",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)