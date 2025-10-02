import streamlit as st
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

from config import (
    FEATURE_TO_COMPONENT_MAP, FEED_MAP, TARGET_RANGES
)

# --- Фичи ---
@st.cache_data
def engineer_features(manual_inputs: dict):
    df = pd.DataFrame([manual_inputs])
    eps = 1e-9
    df['Кукуруза'] = df.get('Кукуруза', 0.0)
    df['Зерновые_прочие'] = df.get('Зерновые_прочие', 0.0)
    df['Комбикорма'] = df.get('Комбикорма', 0.0)
    df['Корнаж_ЗСК'] = df.get('Корнаж_ЗСК', 0.0)
    df['Сенаж'] = df.get('Сенаж', 0.0)
    df['Сено'] = df.get('Сено', 0.0)
    df['Солома'] = df.get('Солома', 0.0)
    df['Шрот_соевый'] = df.get('Шрот_соевый', 0.0)
    df['Шрот_рапсовый'] = df.get('Шрот_рапсовый', 0.0)
    df['Шрот_подсолнечный'] = df.get('Шрот_подсолнечный', 0.0)
    df['Жмых_рапсовый'] = df.get('Жмых_рапсовый', 0.0)
    df['Жмых_льняной'] = df.get('Жмых_льняной', 0.0)
    df['Солома'] = df.get('Солома', 0.0)

    df['Sum_conc'] = df[['Кукуруза', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК']].sum(axis=1)
    df['Sum_rough'] = df[['Сенаж', 'Сено', 'Солома']].sum(axis=1)
    df['Sum_prot_ex'] = df[['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный']].sum(axis=1)
    df['Sum_prot_pr'] = df[['Жмых_рапсовый', 'Жмых_льняной']].sum(axis=1)
    df['Sum_prot'] = df['Sum_prot_ex'] + df['Sum_prot_pr']

    df['share_kukur'] = np.where(df['Sum_conc'] > 0, df['Кукуруза'] / (df['Sum_conc'] + eps), 0.0)
    df['share_korn']  = np.where(df['Sum_conc'] > 0, df['Корнаж_ЗСК'] / (df['Sum_conc'] + eps), 0.0)
    df['share_zern']  = np.where(df['Sum_conc'] > 0, df['Зерновые_прочие'] / (df['Sum_conc'] + eps), 0.0)
    df['share_soloma'] = np.where(df['Sum_rough'] > 0, df['Солома'] / (df['Sum_rough'] + eps), 0.0)
    df['share_prot_soy']  = np.where(df['Sum_prot'] > 0, df['Шрот_соевый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_raps'] = np.where(df['Sum_prot'] > 0, df['Шрот_рапсовый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_lin']  = np.where(df['Sum_prot'] > 0, df['Жмых_льняной'] / (df['Sum_prot'] + eps), 0.0)
    return df

# --- Загрузка моделей + выделение сильных компонентов ---
@st.cache_resource
def load_models_and_get_influencers(paths: dict):
    models = {}
    all_model_features = set()
    for acid_name, path in paths.items():
        if not path.exists():
            st.error(f"Файл модели не найден: {path}")
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
    weak_list   = sorted([c for c in all_components if c not in strong_components])
    return models, strong_list, weak_list

# --- Прогнозы по кислотам ---
def _predict_pack(model_pack: dict, inputs_dict: dict):
    X = engineer_features(inputs_dict)[model_pack['features']]
    if model_pack.get('add_constant', True):
        X = sm.add_constant(X, has_constant='add')
    sf = model_pack['model'].get_prediction(X).summary_frame(alpha=0.05)
    mean = float(max(0.0, sf['mean'].iloc[0]))
    lo   = float(max(0.0, sf['mean_ci_lower'].iloc[0]))
    hi   = float(max(0.0, sf['mean_ci_upper'].iloc[0]))
    return mean, lo, hi

def predict_all_acids(models: dict, inputs_dict: dict, target_ranges: dict):
    predictions = {}
    any_deviations = False

    for acid_name, pack in models.items():
        mean, ci_low, ci_up = _predict_pack(pack, inputs_dict)
        t_min, t_max = target_ranges[acid_name]

        color = '#2ca02c'
        status = "🟢 Норма"
        if mean < t_min or mean > t_max:
            status = "🔴 Вне нормы"; any_deviations = True; color = '#d62728'
        elif ci_low < t_min or ci_up > t_max:
            status = "🟡 Риск отклонения"; any_deviations = True; color = '#ff7f0e'

        predictions[acid_name] = {
            "mean": mean, "ci_lower": ci_low, "ci_upper": ci_up,
            "target_min": t_min, "target_max": t_max,
            "target": f"{t_min:.1f}%–{t_max:.1f}%",
            "status": status, "color": color
        }
    return predictions, any_deviations

# --- Чувствительности ---
def local_sensitivities(acid_name: str, models: dict, base_inputs: dict, step: float = 0.5) -> dict:
    base_mean, _, _ = _predict_pack(models[acid_name], base_inputs)
    sens = {}
    for comp in FEED_MAP.keys():
        x2 = dict(base_inputs)
        x2[comp] = max(0.0, x2.get(comp, 0.0) + step)
        m2, _, _ = _predict_pack(models[acid_name], x2)
        sens[comp] = (m2 - base_mean) / step
    return sens

def sensitivities_matrix(models: dict, base_inputs: dict, step: float = 0.5) -> pd.DataFrame:
    acids = list(models.keys())
    comps = list(FEED_MAP.keys())
    data = {acid: {} for acid in acids}
    for acid in acids:
        s = local_sensitivities(acid, models, base_inputs, step=step)
        for comp in comps:
            data[acid][comp] = s.get(comp, 0.0)
    return pd.DataFrame(data, index=comps)

# --- Меры регулирования ---
def _total_sv(inputs: dict) -> float:
    return float(sum(inputs.values()))

def build_measures(preds: dict, sens_df: pd.DataFrame, base_inputs: dict,
                   sv_bounds=(15.0, 30.0), sv_margin: float = 0.5, top_k: int = 3, tol: float = 1e-6):
    sv = _total_sv(base_inputs)
    sv_min, sv_max = sv_bounds
    allow_inc_global = sv < (sv_max - sv_margin)
    allow_dec_global = sv > (sv_min + sv_margin)

    measures = {}
    all_inc, all_dec = [], []

    for acid, d in preds.items():
        if acid not in sens_df.columns:
            continue
        status = d['status']
        mean = d['mean']
        lo, hi = d['target_min'], d['target_max']
        if '🟢' in status:
            continue

        col = sens_df[acid]
        center = 0.5 * (lo + hi)

        if mean < lo - tol:
            need_up = True
        elif mean > hi + tol:
            need_up = False
        else:
            need_up = mean < center

        if need_up:
            inc_candidates = col[col > 0].sort_values(ascending=False)
            dec_candidates = col[col < 0].abs().sort_values(ascending=False)
        else:
            dec_candidates = col[col > 0].sort_values(ascending=False)
            inc_candidates = col[col < 0].abs().sort_values(ascending=False)

        inc_list, dec_list = [], []

        if allow_inc_global:
            for comp in inc_candidates.index:
                inc_list.append(comp)
                if len(inc_list) >= top_k: break

        if allow_dec_global:
            for comp in dec_candidates.index:
                if base_inputs.get(comp, 0.0) > 0.0:
                    dec_list.append(comp)
                if len(dec_list) >= top_k: break

        measures[acid] = {"status": status, "inc": inc_list, "dec": dec_list}
        all_inc.extend(inc_list)
        all_dec.extend(dec_list)

    from collections import Counter
    inc_count = Counter(all_inc)
    dec_count = Counter(all_dec)
    heavy = {
        "inc": {c for c, n in inc_count.items() if n >= 2},
        "dec": {c for c, n in dec_count.items() if n >= 2},
    }
    return measures, heavy
