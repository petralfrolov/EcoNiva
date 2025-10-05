import streamlit as st
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

from config import (
    FEED_MAP, TARGET_RANGES
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
    df['share_korn'] = np.where(df['Sum_conc'] > 0, df['Корнаж_ЗСК'] / (df['Sum_conc'] + eps), 0.0)
    df['share_zern'] = np.where(df['Sum_conc'] > 0, df['Зерновые_прочие'] / (df['Sum_conc'] + eps), 0.0)
    df['share_soloma'] = np.where(df['Sum_rough'] > 0, df['Солома'] / (df['Sum_rough'] + eps), 0.0)
    df['share_prot_soy'] = np.where(df['Sum_prot'] > 0, df['Шрот_соевый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_raps'] = np.where(df['Sum_prot'] > 0, df['Шрот_рапсовый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_lin'] = np.where(df['Sum_prot'] > 0, df['Жмых_льняной'] / (df['Sum_prot'] + eps), 0.0)
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

    all_components = list(FEED_MAP.keys())
    return models, all_components


# --- Прогнозы по кислотам ---
def _predict_pack(model_pack: dict, inputs_dict: dict):
    x = engineer_features(inputs_dict)[model_pack['features']]
    if model_pack.get('add_constant', True):
        x = sm.add_constant(x, has_constant='add')
    sf = model_pack['model'].get_prediction(x).summary_frame(alpha=0.05)
    mean = float(max(0.0, sf['mean'].iloc[0]))
    lo = float(max(0.0, sf['mean_ci_lower'].iloc[0]))
    hi = float(max(0.0, sf['mean_ci_upper'].iloc[0]))
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
            status = "🔴 Вне нормы"
            any_deviations = True
            color = '#d62728'
        elif ci_low < t_min or ci_up > t_max:
            status = "🟡 Риск отклонения"
            any_deviations = True
            color = '#ff7f0e'

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


def _targets_and_weights(preds: dict, target_ranges: dict):
    acids = list(preds.keys())
    centers = []
    weights = []
    for a in acids:
        lo, hi = target_ranges[a]
        c = 0.5 * (lo + hi)
        centers.append(c)

        st_label = preds[a]["status"]
        mean_val = preds[a]["mean"]  # Получаем текущее среднее значение

        # НОВОЕ ПРАВИЛО: Если мы уже в зеленой зоне, не трогаем
        if "🟢" in st_label and lo <= mean_val <= hi:
            w = 0.0  # Вес равен нулю, цель достигнута
        elif "🔴" in st_label:
            w = 3.0
        elif "🟡" in st_label:
            w = 1.5
        else:
            w = 0.5  # Для случаев, когда статус зеленый, но мы на самой границе
        weights.append(w)

    return np.array(centers, float), np.array(weights, float), acids

def _enforce_sv_bounds(x: np.ndarray, free_mask: np.ndarray, sv_min: float, sv_max: float):
    s = float(x.sum())
    if s > sv_max:
        over = s - sv_max
        free_sum = float(x[free_mask].sum())
        if free_sum > 0:
            factor = max((free_sum - over) / free_sum, 0.0)
            x[free_mask] *= factor
    elif s < sv_min:
        deficit = sv_min - s
        # распределяем дефицит пропорционально текущим массам + 1 (чтоб нули тоже росли)
        w = x[free_mask] + 1.0
        wsum = float(w.sum())
        if wsum > 0:
            x[free_mask] += deficit * (w / wsum)
    x[x < 0] = 0.0
    return x

def optimize_ration(models: dict,
                    base_inputs: dict,
                    target_ranges: dict,
                    locks: set[str] | None = None,
                    sv_bounds=(15.0, 30.0),
                    lock_sv_total: bool = False,
                    step: float = 0.5,
                    lambda_l2: float = 1e-3,
                    max_iter: int = 10,
                    max_delta_per_iter: float = 1):
    """
    Возвращает (new_inputs: dict, report: dict)
    """
    locks = locks or set()
    comps = list(FEED_MAP.keys())
    acids = list(models.keys())

    x = np.array([float(base_inputs.get(c, 0.0)) for c in comps], float)
    free_mask = np.array([c not in locks for c in comps], dtype=bool)
    if not free_mask.any():
        return base_inputs, {"success": False, "reason": "Нет свободных переменных (все залочены)."}

    history = []
    for it in range(1, max_iter + 1):
        # Текущие предсказания
        cur_inputs_dict = {c: float(v) for c, v in zip(comps, x)}
        preds, _ = predict_all_acids(models, cur_inputs_dict, target_ranges)
        y = np.array([preds[a]["mean"] for a in acids], float)
        centers, weights, acids_order = _targets_and_weights(preds, target_ranges)
        r = centers - y  # куда хотим сдвинуть

        # Якоби: dY/dX (acids x comps)
        J_df = sensitivities_matrix(models, cur_inputs_dict, step=step).T
        # гарантируем порядок
        J_df = J_df.reindex(index=acids, columns=comps)
        J = J_df.to_numpy(dtype=float)

        # Оставляем только свободные колонки
        A = J[:, free_mask]
        if A.size == 0:
            return cur_inputs_dict, {"success": False, "reason": "Нет свободных переменных."}

        # Взвешивание по важности целей
        w_sqrt = np.sqrt(weights)
        Aw = (A.T * w_sqrt).T
        rw = r * w_sqrt

        # Ridge-LS: (A^T A + λI) Δ = A^T r
        ATA = Aw.T @ Aw
        ATb = Aw.T @ rw
        ATA.flat[::ATA.shape[0]+1] += lambda_l2  # +λI на диагональ
        try:
            delta_free = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            delta_free, *_ = np.linalg.lstsq(ATA, ATb, rcond=None)

        if lock_sv_total and len(delta_free) > 0:
            # Корректируем вектор изменений так, чтобы его сумма была равна нулю.
            # Это гарантирует, что сумма компонентов рациона не изменится.
            delta_free -= delta_free.mean()

        # Ограничение шага за итерацию
        delta_free = np.clip(delta_free, -max_delta_per_iter, max_delta_per_iter)

        # Собираем полный вектор Δx
        delta = np.zeros_like(x)
        delta[free_mask] = delta_free

        # Применяем и проекции ограничений
        if lock_sv_total:
            s_target = x.sum()  # Запоминаем целевую сумму
            x = x + delta
            x[x < 0] = 0.0  # Обрезаем отрицательные значения

            s_current = x[free_mask].sum()  # Считаем сумму только свободных компонентов
            s_locked = x[~free_mask].sum()  # и заблокированных

            # Рассчитываем новый множитель для свободных компонентов
            if s_current > 1e-6:
                factor = (s_target - s_locked) / s_current
                x[free_mask] *= factor

            x[x < 0] = 0.0  # Финальная проверка на всякий случай
        else:
            # Старая логика для режима без блокировки СВ
            x = x + delta
            x[x < 0] = 0.0
            x = _enforce_sv_bounds(x, free_mask, sv_bounds[0], sv_bounds[1])

        # Сохраняем прогресс
        history.append({
            "iter": it,
            "resid_norm": float(np.linalg.norm(r, 2)),
            "sum_sv": float(x.sum())
        })

        # Проверка «зелёной зоны»
        cur_inputs_dict = {c: float(v) for c, v in zip(comps, x)}
        preds2, _ = predict_all_acids(models, cur_inputs_dict, target_ranges)
        ok_all = True
        out_list = []
        for a in acids:
            lo, hi = target_ranges[a]
            m = preds2[a]["mean"]
            ci_lower_val = preds2[a]["ci_lower"]
            ci_upper_val = preds2[a]["ci_upper"]
            if not (ci_lower_val > lo and ci_upper_val < hi):
                ok_all = False
                out_list.append((a, m, lo, hi))
        if ok_all:
            base_vec = np.array([float(base_inputs.get(c, 0.0)) for c in comps], float)
            return cur_inputs_dict, {
                "success": True,
                "iterations": it,
                "delta_total": float((x - base_vec).sum()),
                "deltas": {c: float(xi - bi) for c, xi, bi in zip(comps, x, base_vec)},
                "history": history,
                "out_of_range": []
            }

    # Если не уложились, возвращаем лучшее найденное
    base_vec = np.array([float(base_inputs.get(c, 0.0)) for c in comps], float)
    cur_inputs_dict = {c: float(v) for c, v in zip(comps, x)}
    preds2, _ = predict_all_acids(models, cur_inputs_dict, target_ranges)
    out_list = []
    for a in acids:
        lo, hi = target_ranges[a]
        m = preds2[a]["mean"]
        if not (lo <= m <= hi):
            out_list.append((a, m, lo, hi))
    return cur_inputs_dict, {
        "success": False,
        "iterations": max_iter,
        "delta_total": float((x - base_vec).sum()),
        "deltas": {c: float(xi - bi) for c, xi, bi in zip(comps, x, base_vec)},
        "history": history,
        "out_of_range": out_list
    }
