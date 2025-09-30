import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import pickle
import statsmodels.api as sm
import plotly.graph_objects as go
from pathlib import Path

# --- КОНФИГУРАЦИЯ И КОНСТАНТЫ (без изменений) ---
st.set_page_config(layout="wide", page_title="Анализатор рациона коров")
# ... (весь блок FEED_MAP, MODEL_PATHS, TARGET_RANGES, FEATURE_TO_COMPONENT_MAP остается без изменений)
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
MODEL_PATHS = {
    "Лауриновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Лауриновая.pkl'),
    "Пальмитиновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Пальмитиновая.pkl'),
    "Стеариновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Стеариновая.pkl'),
    "Олеиновая": Path(r'C:\Users\Петр\papka bebrapka\ols_Олеиновая.pkl'),
}
TARGET_RANGES = {
    "Лауриновая": (2.0, 4.4), "Пальмитиновая": (21.0, 32.0),
    "Стеариновая": (8.0, 13.5), "Олеиновая": (20.0, 28.0),
}
FEATURE_TO_COMPONENT_MAP = {
    'Sum_conc': ['Кукуруза', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК'],
    'Sum_rough': ['Сенаж', 'Сено', 'Солома'],
    'Sum_prot': ['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный', 'Жмых_рапсовый', 'Жмых_льняной'],
    'share_kukur': ['Кукуруза'], 'share_korn': ['Корнаж_ЗСК'], 'share_zern': ['Зерновые_прочие'],
    'share_soloma': ['Солома'], 'share_prot_soy': ['Шрот_соевый'], 'share_prot_raps': ['Шрот_рапсовый'],
    'share_prot_lin': ['Жмых_льняной'], 'Силос': ['Силос'], 'Жиры': ['Жиры'],
    'Побочные_свекловичные': ['Побочные_свекловичные'], 'Побочные_прочие': ['Побочные_прочие']
}


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (без изменений) ---
@st.cache_data
def parse_pdf_report(uploaded_file):
    logs = []
    if uploaded_file is None: return {}, ["Файл не был загружен."]
    logs.append(f"Начало обработки файла: {uploaded_file.name}")
    tables = []
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                extracted = page.extract_tables()
                if extracted: tables.extend(extracted)
        logs.append(f"Найдено таблиц в PDF: {len(tables)}")
    except Exception as e:
        logs.append(f"КРИТИЧЕСКАЯ ОШИБКА при чтении PDF: {e}");
        return {}, logs
    if not tables or len(tables[0]) < 2:
        logs.append("ОШИБКА: Не удалось найти подходящую таблицу с данными в PDF.");
        return {}, logs
    df = pd.DataFrame(tables[0][1:], columns=tables[0][0])
    df.columns = df.iloc[0];
    df = df[1:].reset_index(drop=True)
    logs.append(f"Обнаружены колонки: {list(df.columns)}")
    ingredient_col = next((c for c in df.columns if 'Ингредиент' in c), None)
    sv_kg_col = next((c for c in df.columns if 'СВ кг' in c), None)
    if not ingredient_col or not sv_kg_col:
        logs.append("ОШИБКА: В таблице не найдены обязательные колонки 'Ингредиенты' или 'СВ кг'.");
        return {}, logs
    logs.append(f"Колонка ингредиентов: '{ingredient_col}', колонка данных: '{sv_kg_col}'")
    df = df[[ingredient_col, sv_kg_col]].copy()
    df[sv_kg_col] = pd.to_numeric(df[sv_kg_col].str.replace(',', '.'), errors='coerce')
    df.dropna(inplace=True);
    logs.append(f"Найдено {len(df)} строк с числовыми данными.")
    df['clean_name'] = (df[ingredient_col].astype(str).str.split('/', n=1).str[0]
                        .str.replace(r'\s{2,}', ' ', regex=True)
                        .str.replace(r'\s*\d.*$', '', regex=True)
                        .str.strip('.,\ '))
    aggregated_data = {key: 0.0 for key in FEED_MAP.keys()}
    logs.append("\n--- Построчное сопоставление (первые 15 строк) ---")
    for index, row in df.head(15).iterrows():
        match_found = False
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += row[sv_kg_col]
                logs.append(
                    f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✓ '{category}' ({row[sv_kg_col]:.2f} кг)")
                match_found = True
                break
        if not match_found:
            logs.append(
                f"Строка {index + 1}: '{row[ingredient_col]}' -> '{row['clean_name']}' -> ✗ Категория не найдена")
    for index, row in df.iloc[15:].iterrows():
        for category, names in FEED_MAP.items():
            if any(name.strip().lower() in row['clean_name'].lower() for name in names):
                aggregated_data[category] += row[sv_kg_col]
                break
    logs.append("\n--- Итог агрегации ---")
    logs.append(str({k: round(v, 2) for k, v in aggregated_data.items() if v > 0}))
    if sum(aggregated_data.values()) < 0.01:
        logs.append("ПРЕДУПРЕЖДЕНИЕ: Сумма всех найденных компонентов равна нулю. Данные не загружены.")
        return {}, logs
    return aggregated_data, logs


@st.cache_data
def engineer_features(manual_inputs: dict):
    df = pd.DataFrame([manual_inputs]);
    eps = 1e-9
    df['Sum_conc'] = df[['Кукуруза', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК']].sum(axis=1)
    df['Sum_rough'] = df[['Сенаж', 'Сено', 'Солома']].sum(axis=1)
    df['Sum_prot_ex'] = df[['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный']].sum(axis=1)
    df['Sum_prot_pr'] = df[['Жмых_рапсовый', 'Жмых_льняной']].sum(axis=1)
    df['Sum_prot'] = df['Sum_prot_ex'] + df['Sum_prot_pr']
    df['share_kukur'] = np.where(df['Sum_conc'] > 0, df['Кукуруза'] / (df['Sum_conc'] + eps), 0.0)
    df['share_korn'] = np.where(df['Sum_conc'] > 0, df['Корнаж_ЗСК'] / (df['Sum_conc'] + eps), 0.0)
    df['share_zern'] = np.where(df['Sum_conc'] > 0, df['Зерновые_прочие'] / (df['Sum_conc'] + eps), 0.0)
    df['share_soloma'] = np.where(df['Sum_rough'] > 0, df['Солома'] / (df['Sum_rough'] + eps), 0.0)
    df['share_prot_soy'] = np.where(df['Sum_prot'] > 0, df['Шрот_соевый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_raps'] = np.where(df['Sum_prot'] > 0, df['Шрот_рапсовый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_lin'] = np.where(df['Sum_prot'] > 0, df['Жмых_льняной'] / (df['Sum_prot'] + eps), 0.0)
    return df


@st.cache_resource
def load_models_and_get_influencers(paths: dict):
    models = {};
    all_model_features = set()
    for acid_name, path in paths.items():
        if not path.exists():
            st.error(f"Файл модели не найден: {path}");
            return None, [], []
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            models[acid_name] = model_data
            all_model_features.update(model_data['features'])
    strong_components = set()
    for feature in all_model_features:
        if feature in FEATURE_TO_COMPONENT_MAP:
            strong_components.update(FEATURE_TO_COMPONENT_MAP[feature])
    all_components = list(FEED_MAP.keys())
    strong_list = sorted([c for c in all_components if c in strong_components])
    weak_list = sorted([c for c in all_components if c not in strong_components])
    return models, strong_list, weak_list


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---

def run_analysis():
    current_inputs = {key: st.session_state[f"inp_{key}"] for key in FEED_MAP.keys()}

    # Сохраняем текущие инпуты и общую сумму в состояние для отображения
    st.session_state.current_inputs = current_inputs
    st.session_state.total_sv = sum(current_inputs.values())

    features_df = engineer_features(current_inputs)

    predictions = {}
    any_deviations = False
    for acid_name, model_data in models.items():
        model = model_data['model'];
        model_features = model_data['features']
        if not all(f in features_df.columns for f in model_features):
            st.error(f"Ошибка: для модели '{acid_name}' не хватает признаков.");
            continue
        X = features_df[model_features]
        if model_data.get('add_constant', True): X = sm.add_constant(X, has_constant='add')
        pred_summary = model.get_prediction(X).summary_frame(alpha=0.05)
        mean, ci_low, ci_up = pred_summary['mean'].iloc[0], pred_summary['mean_ci_lower'].iloc[0], \
        pred_summary['mean_ci_upper'].iloc[0]
        t_min, t_max = TARGET_RANGES[acid_name]
        color = '#2ca02c';
        status = "🟢 Норма"
        if mean < t_min or mean > t_max:
            status = "🔴 Вне нормы";
            any_deviations = True;
            color = '#d62728'
        elif ci_low < t_min or ci_up > t_max:
            status = "🟡 Риск отклонения";
            any_deviations = True;
            color = '#ff7f0e'
        predictions[acid_name] = {"mean": mean, "ci_lower": ci_low, "ci_upper": ci_up,
                                  "target": f"{t_min:.1f}%–{t_max:.1f}%", "target_min": t_min, "target_max": t_max,
                                  "status": status, "color": color}

    st.session_state.predictions = predictions
    st.session_state.any_deviations = any_deviations
    st.session_state.analysis_run = True


def reset_app_state():
    for key in FEED_MAP.keys():
        st.session_state[f"inp_{key}"] = 0.0
    st.session_state.analysis_run = False
    st.session_state.logs = []
    st.session_state.last_uploaded_filename = None
    if 'current_inputs' in st.session_state:
        del st.session_state['current_inputs']


# Инициализация состояния
if 'analysis_run' not in st.session_state: st.session_state.analysis_run = False
if 'logs' not in st.session_state: st.session_state.logs = []
models, strong_influencers, weak_influencers = load_models_and_get_influencers(MODEL_PATHS)
if models is None: st.stop()
for key in FEED_MAP.keys():
    if f"inp_{key}" not in st.session_state: st.session_state[f"inp_{key}"] = 0.0

with st.sidebar:
    st.header("⚙️ Параметры рациона")
    uploaded_file = st.file_uploader("Загрузите PDF-отчет", type="pdf")

    if uploaded_file is not None:
        if st.session_state.get('last_uploaded_filename') != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            parsed_data, logs = parse_pdf_report(uploaded_file)
            st.session_state.logs = logs
            if parsed_data:
                for key, value in parsed_data.items():
                    st.session_state[f"inp_{key}"] = value
                run_analysis()
                st.rerun()

    st.subheader("Состав рациона (кг СВ):")
    st.markdown("**Сильно влияющие компоненты**")
    for key in strong_influencers:
        st.number_input(key, min_value=0.0, step=0.1, format="%.2f", key=f"inp_{key}", on_change=run_analysis)

    st.markdown("**Прочие компоненты**")
    for key in weak_influencers:
        st.number_input(key, min_value=0.0, step=0.1, format="%.2f", key=f"inp_{key}", on_change=run_analysis)

    col1, col2 = st.columns(2)
    col1.button("📈 Рассчитать", use_container_width=True, type="primary", on_click=run_analysis)
    col2.button("Сбросить", use_container_width=True, on_click=reset_app_state)

# --- ОТОБРАЖЕНИЕ РЕЗУЛЬТАТОВ ---
st.title("🐄 Аналитический дашборд")

if st.session_state.logs:
    with st.expander("📝 Логи разбора PDF-файла", expanded=False):
        st.code("\n".join(st.session_state.logs), language='text')

if not st.session_state.analysis_run:
    st.info("Введите данные в панели слева или загрузите PDF-отчет для начала анализа.")
else:
    # --- НОВЫЙ БЛОК: РАЦИОН ---
    st.subheader("Рацион")
    col1, col2 = st.columns([2, 1])  # Делаем первую колонку в 2 раза шире

    with col1:
        current_inputs = st.session_state.current_inputs
        pie_data = {k: v for k, v in current_inputs.items() if v > 0}
        if pie_data:
            pie_fig = go.Figure(data=[go.Pie(
                labels=list(pie_data.keys()),
                values=list(pie_data.values()),
                hole=.3
            )])
            pie_fig.update_layout(title_text="Структура рациона по категориям", showlegend=False)
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("Нет данных для отображения структуры рациона.")

    with col2:
        total_sv = st.session_state.total_sv
        st.metric(label="Сумма СВ кг/день", value=f"{total_sv:.2f}")

        if total_sv < 15:
            st.error("🔴 Общее количество СВ ниже нормы (< 15 кг). Рекомендуется увеличить объем рациона.")
        elif total_sv > 30:
            st.error("🔴 Общее количество СВ выше нормы (> 30 кг). Рекомендуется снизить объем рациона.")
        else:
            st.success("🟢 Общее количество СВ в норме.")

    st.markdown("---")
    preds = st.session_state.predictions
    st.subheader("Прогноз по жирным кислотам")
    cols = st.columns(len(preds))
    # ... (отображение метрик и рекомендаций без изменений)
    for i, (acid_name, data) in enumerate(preds.items()):
        cols[i].metric(
            label=f"{acid_name} ({data['status']})", value=f"{data['mean']:.2f}%",
            help=f"Цель: {data['target']}. 95% ДИ: {data['ci_lower']:.2f}%–{data['ci_upper']:.2f}%"
        )
    if st.session_state.any_deviations:
        recs = [f"**{acid}**: {data['status']}. Проверьте влияние связанных компонентов." for acid, data in
                preds.items() if "🟢" not in data['status']]
        st.warning("\n".join(f"* {rec}" for rec in recs))

    # --- ИСПРАВЛЕННЫЙ БЛОК ВИЗУАЛИЗАЦИИ ---
    fig = go.Figure()

    acid_names = list(preds.keys())
    mean_values = [p['mean'] for p in preds.values()]

    # Сначала добавляем цветные бары
    for acid_name in reversed(acid_names):
        p = preds[acid_name]
        fig.add_trace(go.Bar(
            y=[acid_name], x=[p['mean']], name=acid_name, orientation='h', marker_color=p['color'],
            error_x=dict(type='data', symmetric=False, array=[p['ci_upper'] - p['mean']],
                         arrayminus=[p['mean'] - p['ci_lower']], thickness=1.5, width=4),
            text=f"{p['mean']:.2f}%", textposition='outside'
        ))

    # Потом добавляем зоны, чтобы они были сверху
    for acid_name in reversed(acid_names):
        p = preds[acid_name]
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=p['target_min'], y0=acid_name,
            x1=p['target_max'], y1=acid_name,
            y0shift=-0.35, y1shift=0.35,
            fillcolor="green", opacity=0.2,  # Полупрозрачный цвет
            layer="above",  # Отображаем поверх баров
            line_width=0
        )

    # В самом конце добавляем жирные точки прогноза
    fig.add_trace(go.Scatter(
        y=acid_names,
        x=mean_values,
        mode='markers',
        marker=dict(color='black', size=8, symbol='diamond'),
        showlegend=False
    ))

    fig.update_layout(
        title_text="Прогноз (◆), 95% ДИ (линия) и целевой диапазон (зеленая зона)",
        barmode='stack', yaxis_title="Жирная кислота", xaxis_title="Содержание, %",
        showlegend=False, height=400
    )
    st.plotly_chart(fig, use_container_width=True)