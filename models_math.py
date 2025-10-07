import streamlit as st
import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm

from config import (
    FEED_MAP, TARGET_RANGES, TARGET_COLS
)


@st.cache_data
def engineer_features(manual_inputs: dict):
    """
    Создает производные признаки (фичи) на основе базового состава рациона.

    На входе получает словарь с массами базовых компонентов и рассчитывает
    суммарные и долевые показатели (например, сумма концентратов,
    доля кукурузы в концентратах), которые используются в моделях.

    Args:
        manual_inputs (dict): Словарь, где ключи - названия групп кормов,
                              а значения - их масса в кг СВ.

    Returns:
        pd.DataFrame: DataFrame с одной строкой, содержащей все
                      исходные и производные признаки.
    """
    df = pd.DataFrame([manual_inputs])
    eps = 1e-9
    # Гарантируем наличие всех колонок, даже если их значение 0
    for key in FEED_MAP.keys():
        if key not in df.columns:
            df[key] = 0.0

    df['Sum_conc'] = df[['Кукуруза_сухая', 'Кукуруза_влажная', 'Зерновые_прочие', 'Комбикорма', 'Корнаж_ЗСК']].sum(
        axis=1)
    df['Sum_rough'] = df[['Сенаж', 'Сено', 'Солома']].sum(axis=1)
    df['Sum_prot_ex'] = df[['Шрот_соевый', 'Шрот_рапсовый', 'Шрот_подсолнечный']].sum(axis=1)
    df['Sum_prot_pr'] = df[['Жмых_рапсовый', 'Жмых_льняной']].sum(axis=1)
    df['Sum_prot'] = df['Sum_prot_ex'] + df['Sum_prot_pr']

    df['share_kukur_wet'] = np.where(df['Sum_conc'] > 0, (df['Кукуруза_влажная']) / (df['Sum_conc'] + eps), 0.0)
    df['share_kukur_dry'] = np.where(df['Sum_conc'] > 0, (df['Кукуруза_сухая']) / (df['Sum_conc'] + eps), 0.0)
    df['share_korn'] = np.where(df['Sum_conc'] > 0, df['Корнаж_ЗСК'] / (df['Sum_conc'] + eps), 0.0)
    df['share_zern'] = np.where(df['Sum_conc'] > 0, df['Зерновые_прочие'] / (df['Sum_conc'] + eps), 0.0)
    df['share_soloma'] = np.where(df['Sum_rough'] > 0, df['Солома'] / (df['Sum_rough'] + eps), 0.0)
    df['share_prot_soy'] = np.where(df['Sum_prot'] > 0, df['Шрот_соевый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_raps'] = np.where(df['Sum_prot'] > 0, df['Шрот_рапсовый'] / (df['Sum_prot'] + eps), 0.0)
    df['share_prot_lin'] = np.where(df['Sum_prot'] > 0, df['Жмых_льняной'] / (df['Sum_prot'] + eps), 0.0)
    return df


@st.cache_resource
def load_models_and_get_influencers(paths: dict):
    """
    Загружает сериализованные модели регрессии из файлов pickle.

    Кэшируется как ресурс Streamlit для предотвращения повторной загрузки.
    Также собирает уникальный список всех признаков, используемых в моделях.

    Args:
        paths (dict): Словарь, где ключи - названия кислот, а значения -
                      пути к файлам моделей.

    Returns:
        tuple: Кортеж из двух элементов:
               - dict: Словарь с загруженными объектами моделей.
               - list: Список всех уникальных названий компонентов рациона.
    """
    models = {}
    all_model_features = set()
    for acid_name, path in paths.items():
        if not path.exists():
            st.error(f"Файл модели не найден: {path}")
            return None, []
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            models[acid_name] = model_data
            all_model_features.update(model_data['features'])

    all_components = list(FEED_MAP.keys())
    return models, all_components


def _predict_pack(model_pack: dict, inputs_dict: dict):
    """
    Внутренняя функция для получения прогноза от одной модели.

    Args:
        model_pack (dict): Словарь, содержащий объект модели и ее метаданные.
        inputs_dict (dict): Словарь с входными данными для предсказания.

    Returns:
        tuple: Среднее прогнозное значение, нижняя и верхняя границы 95%
               доверительного интервала.
    """
    x = engineer_features(inputs_dict)[model_pack['features']]
    if model_pack.get('add_constant', True):
        x = sm.add_constant(x, has_constant='add')
    sf = model_pack['model'].get_prediction(x).summary_frame(alpha=0.05)
    mean = float(max(0.0, sf['mean'].iloc[0]))
    lo = float(max(0.0, sf['mean_ci_lower'].iloc[0]))
    hi = float(max(0.0, sf['mean_ci_upper'].iloc[0]))
    return mean, lo, hi


def predict_all_acids(models: dict, inputs_dict: dict, target_ranges: dict):
    """
    Выполняет прогнозирование для всех жирных кислот.

    Итерируется по всем загруженным моделям, получает прогноз для каждой,
    сравнивает результат с целевым диапазоном и формирует статус
    ("Норма", "Риск отклонения", "Вне нормы").

    Args:
        models (dict): Словарь с загруженными моделями.
        inputs_dict (dict): Словарь с текущим составом рациона.
        target_ranges (dict): Словарь с целевыми диапазонами для каждой кислоты.

    Returns:
        tuple: Кортеж из двух элементов:
               - dict: Словарь с подробными результатами прогноза для каждой кислоты.
               - bool: Флаг, указывающий, есть ли хотя бы одно отклонение от нормы.
    """
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


def local_sensitivities(acid_name: str, models: dict, base_inputs: dict, step: float = 0.5) -> dict:
    """
    Рассчитывает локальные чувствительности (частные производные) для одной кислоты.

    Оценивает, как изменится прогнозное содержание одной жирной кислоты
    при увеличении массы каждого компонента рациона на `step` кг СВ.

    Args:
        acid_name (str): Название жирной кислоты для анализа.
        models (dict): Словарь с загруженными моделями.
        base_inputs (dict): Базовый состав рациона.
        step (float): Величина изменения массы компонента для расчета производной.

    Returns:
        dict: Словарь, где ключи - названия компонентов, а значения -
              рассчитанная чувствительность.
    """
    base_mean, _, _ = _predict_pack(models[acid_name], base_inputs)
    sens = {}
    for comp in FEED_MAP.keys():
        x2 = dict(base_inputs)
        x2[comp] = max(0.0, x2.get(comp, 0.0) + step)
        m2, _, _ = _predict_pack(models[acid_name], x2)
        sens[comp] = (m2 - base_mean) / step
    return sens


def sensitivities_matrix(models: dict, base_inputs: dict, step: float = 0.5) -> pd.DataFrame:
    """
    Строит матрицу чувствительностей (Якобиан) для всех кислот и компонентов.

    Вызывает `local_sensitivities` для каждой кислоты и объединяет
    результаты в единый DataFrame.

    Args:
        models (dict): Словарь с загруженными моделями.
        base_inputs (dict): Базовый состав рациона.
        step (float): Величина изменения для расчета производных.

    Returns:
        pd.DataFrame: Матрица чувствительностей, где строки - компоненты,
                      а столбцы - жирные кислоты.
    """
    acids = list(models.keys())
    comps = list(FEED_MAP.keys())
    data = {acid: {} for acid in acids}
    for acid in acids:
        s = local_sensitivities(acid, models, base_inputs, step=step)
        for comp in comps:
            data[acid][comp] = s.get(comp, 0.0)
    return pd.DataFrame(data, index=comps)


def _total_sv(inputs: dict) -> float:
    """Простая утилита для расчета общего сухого вещества в рационе."""
    return float(sum(inputs.values()))


def build_measures(preds: dict, sens_df: pd.DataFrame, base_inputs: dict,
                   sv_bounds=(15.0, 30.0), sv_margin: float = 0.5, top_k: int = 3, tol: float = 1e-6):
    """
    Формирует текстовые рекомендации по корректировке рациона.

    Для каждой кислоты с отклонением от нормы, на основе матрицы
    чувствительностей, определяет `top_k` наиболее эффективных
    компонентов для увеличения и уменьшения, чтобы вернуть показатель
    в целевой диапазон.

    Args:
        preds (dict): Результаты прогнозирования.
        sens_df (pd.DataFrame): Матрица чувствительностей.
        base_inputs (dict): Текущий состав рациона.
        sv_bounds (tuple): Границы нормы общего СВ.
        sv_margin (float): "Буфер" от границ СВ, чтобы избежать крайних рекомендаций.
        top_k (int): Количество рекомендуемых компонентов для изменения.
        tol (float): Допуск для определения отклонения.

    Returns:
        tuple: Кортеж из двух словарей:
               - measures: Рекомендации по каждой кислоте.
               - heavy: Компоненты, которые встречаются в рекомендациях наиболее часто.
    """
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
                if len(inc_list) >= top_k:
                    break

        if allow_dec_global:
            for comp in dec_candidates.index:
                if base_inputs.get(comp, 0.0) > 0.0:
                    dec_list.append(comp)
                if len(dec_list) >= top_k:
                    break

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
    """
    Вспомогательная функция для оптимизатора.

    Определяет целевые значения (центры диапазонов) и веса для каждой
    кислоты. Вес тем выше, чем сильнее отклонение от нормы.

    Args:
        preds (dict): Текущие прогнозы.
        target_ranges (dict): Целевые диапазоны.

    Returns:
        tuple: np.array с центрами, np.array с весами, список кислот.
    """
    acids = list(preds.keys())
    centers = []
    weights = []
    for a in acids:
        lo, hi = target_ranges[a]
        c = 0.5 * (lo + hi)
        centers.append(c)

        st_label = preds[a]["status"]
        mean_val = preds[a]["mean"]

        if "🟢" in st_label and lo <= mean_val <= hi:
            w = 0.0
        elif "🔴" in st_label:
            w = 3.0
        elif "🟡" in st_label:
            w = 1.5
        else:
            w = 0.5
        weights.append(w)

    return np.array(centers, float), np.array(weights, float), acids


def _enforce_sv_bounds(x: np.ndarray, free_mask: np.ndarray, sv_min: float, sv_max: float):
    """
    Вспомогательная функция для оптимизатора.

    Корректирует вектор состава рациона так, чтобы общее СВ не выходило
    за заданные границы, изменяя только незаблокированные компоненты.

    Args:
        x (np.ndarray): Вектор состава рациона.
        free_mask (np.ndarray): Маска незаблокированных компонентов.
        sv_min (float): Минимальное общее СВ.
        sv_max (float): Максимальное общее СВ.

    Returns:
        np.ndarray: Скорректированный вектор состава рациона.
    """
    s = float(x.sum())
    if s > sv_max:
        over = s - sv_max
        free_sum = float(x[free_mask].sum())
        if free_sum > 0:
            factor = max((free_sum - over) / free_sum, 0.0)
            x[free_mask] *= factor
    elif s < sv_min:
        deficit = sv_min - s
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
                    max_delta_per_iter: float = 1,
                    round_precision: int = 2):
    """
    Автоматически подбирает оптимальный состав рациона.

    Использует итеративный метод на основе градиентного спуска (решение
    линейной системы с регуляризацией Ridge) для минимизации взвешенной
    суммы квадратов отклонений от целевых значений жирных кислот.
    Итоговый результат округляется до 100 граммов (0.1 кг).

    Args:
        models (dict): Словарь с моделями.
        base_inputs (dict): Исходный рацион.
        target_ranges (dict): Целевые диапазоны.
        locks (set[str] | None): Множество заблокированных компонентов.
        sv_bounds (tuple): Границы общего СВ.
        lock_sv_total (bool): Флаг, нужно ли сохранять общее СВ неизменным.
        step (float): Шаг для расчета Якобиана.
        lambda_l2 (float): Коэффициент L2-регуляризации.
        max_iter (int): Максимальное число итераций.
        max_delta_per_iter (float): Максимальное изменение одного компонента за итерацию.
        round_precision (int): Количество знаков после запятой при округлении.

    Returns:
        tuple: Кортеж из двух элементов:
               - dict: Новый, оптимизированный и округленный состав рациона.
               - dict: Отчет о процессе оптимизации.
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
        cur_inputs_dict = {c: float(v) for c, v in zip(comps, x)}
        preds, _ = predict_all_acids(models, cur_inputs_dict, target_ranges)
        y = np.array([preds[a]["mean"] for a in acids], float)
        centers, weights, acids_order = _targets_and_weights(preds, target_ranges)
        r = centers - y

        # Якоби: dY/dX (acids x comps)
        J_df = sensitivities_matrix(models, cur_inputs_dict, step=step).T
        J_df = J_df.reindex(index=acids, columns=comps)
        J = J_df.to_numpy(dtype=float)

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
        ATA.flat[::ATA.shape[0] + 1] += lambda_l2
        try:
            delta_free = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            delta_free, *_ = np.linalg.lstsq(ATA, ATb, rcond=None)

        if lock_sv_total and len(delta_free) > 0:
            # Корректируем вектор изменений так, чтобы его сумма была равна нулю.
            # Это гарантирует, что сумма компонентов рациона не изменится.
            delta_free -= delta_free.mean()

        delta_free = np.clip(delta_free, -max_delta_per_iter, max_delta_per_iter)

        delta = np.zeros_like(x)
        delta[free_mask] = delta_free

        # Применяем и проекции ограничений
        if lock_sv_total:
            s_target = x.sum()
            x = x + delta
            x[x < 0] = 0.0

            s_current = x[free_mask].sum()
            s_locked = x[~free_mask].sum()

            if s_current > 1e-6:
                factor = (s_target - s_locked) / s_current
                x[free_mask] *= factor
            x[x < 0] = 0.0
        else:
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
        cur_inputs_dict_precise = {c: float(v) for c, v in zip(comps, x)}
        preds2, _ = predict_all_acids(models, cur_inputs_dict_precise, target_ranges)
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
            x = np.round(x, round_precision)
            cur_inputs_dict = {c: float(v) for c, v in zip(comps, x)}
            return cur_inputs_dict, {
                "success": True, "iterations": it,
                "delta_total": float((x - base_vec).sum()),
                "deltas": {c: float(xi - bi) for c, xi, bi in zip(comps, x, base_vec)},
                "history": history, "out_of_range": []
            }

    # Если не уложились, возвращаем лучшее найденное
    base_vec = np.array([float(base_inputs.get(c, 0.0)) for c in comps], float)
    x = np.round(x, round_precision)
    cur_inputs_dict = {c: float(v) for c, v in zip(comps, x)}
    preds2, _ = predict_all_acids(models, cur_inputs_dict, target_ranges)
    out_list = []
    for a in acids:
        lo, hi = target_ranges[a]
        m = preds2[a]["mean"]
        if not (lo <= m <= hi):
            out_list.append((a, m, lo, hi))

    return cur_inputs_dict, {
        "success": False, "iterations": max_iter,
        "delta_total": float((x - base_vec).sum()),
        "deltas": {c: float(xi - bi) for c, xi, bi in zip(comps, x, base_vec)},
        "history": history, "out_of_range": out_list
    }


@st.cache_data
def engineer_features_nutrients(nutrients_inputs: dict, ensure_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Готовит фичи для моделей, обученных на нутриентах.

    Берёт словарь {нутриент: значение} и гарантирует наличие всех колонок,
    необходимых модели (отсутствующие -> 0.0).

    Args:
        nutrients_inputs (dict): значения нутриентов.
        ensure_cols (list[str] | None): явный список колонок; если None — используем TARGET_COLS.

    Returns:
        pd.DataFrame: одна строка с признаками для предсказания.
    """
    cols = ensure_cols or list(TARGET_COLS)
    row = {c: float(nutrients_inputs.get(c, 0.0)) for c in cols}
    return pd.DataFrame([row])[cols]


def _predict_pack_nutrients(model_pack: dict, nutrients_inputs: dict):
    """
    Внутренний предиктор для «нутриентных» моделей.

    Args:
        model_pack (dict): {'model', 'features', 'add_constant'?}
        nutrients_inputs (dict): словарь значений нутриентов.

    Returns:
        tuple: (mean, ci_low, ci_up) — прогноз и 95% ДИ
    """
    X = engineer_features_nutrients(nutrients_inputs, ensure_cols=model_pack['features'])
    if model_pack.get('add_constant', True):
        X = sm.add_constant(X, has_constant='add')
    sf = model_pack['model'].get_prediction(X).summary_frame(alpha=0.05)
    mean = float(max(0.0, sf['mean'].iloc[0]))
    lo = float(max(0.0, sf['mean_ci_lower'].iloc[0]))
    hi = float(max(0.0, sf['mean_ci_upper'].iloc[0]))
    return mean, lo, hi


def predict_all_acids_from_nutrients(models_nutri: dict, nutrients_inputs: dict, target_ranges: dict):
    """
    Прогноз всех кислот из набора нутриентов (вторая вкладка).

    Args:
        models_nutri (dict): модели, обученные на нутриентах.
        nutrients_inputs (dict): значения нутриентов.
        target_ranges (dict): целевые диапазоны по кислотам.

    Returns:
        tuple[dict, bool]: как в predict_all_acids — словарь метрик и флаг отклонений.
    """
    predictions, any_deviations = {}, False
    for acid_name, pack in models_nutri.items():
        mean, ci_low, ci_up = _predict_pack_nutrients(pack, nutrients_inputs)
        t_min, t_max = target_ranges[acid_name]

        color, status = '#2ca02c', "🟢 Норма"
        if mean < t_min or mean > t_max:
            status, any_deviations, color = "🔴 Вне нормы", True, '#d62728'
        elif ci_low < t_min or ci_up > t_max:
            status, any_deviations, color = "🟡 Риск отклонения", True, '#ff7f0e'

        predictions[acid_name] = {
            "mean": mean, "ci_lower": ci_low, "ci_upper": ci_up,
            "target_min": t_min, "target_max": t_max, "target": f"{t_min:.1f}%–{t_max:.1f}%",
            "status": status, "color": color
        }
    return predictions, any_deviations


def local_sensitivities_nutrients(acid_name: str, models_nutri: dict, base_nutrients: dict, step: float = 1.0) -> dict:
    """
    Локальные чувствительности d(кислота)/d(нутриент) при увеличении нутриента на `step`.

    Args:
        acid_name (str): целевая кислота.
        models_nutri (dict): «нутриентные» модели.
        base_nutrients (dict): текущие значения нутриентов.
        step (float): шаг изменения нутриента (в его единицах).

    Returns:
        dict[str,float]: {нутриент: чувствительность}.
    """
    base_mean, _, _ = _predict_pack_nutrients(models_nutri[acid_name], base_nutrients)
    sens = {}
    # считаем по всем целевым колонкам, чтобы таблица была полной
    for col in TARGET_COLS:
        x2 = dict(base_nutrients)
        x2[col] = max(0.0, float(x2.get(col, 0.0)) + step)
        m2, _, _ = _predict_pack_nutrients(models_nutri[acid_name], x2)
        sens[col] = (m2 - base_mean) / step
    return sens


def sensitivities_matrix_nutrients(models_nutri: dict, base_nutrients: dict, step: float = 1.0) -> pd.DataFrame:
    """
    Матрица чувствительностей (Якобиан) dY/dZ для кислот Y по нутриентам Z.

    Args:
        models_nutri (dict): «нутриентные» модели.
        base_nutrients (dict): значения нутриентов.
        step (float): шаг изменения нутриента.

    Returns:
        pd.DataFrame: строки — нутриенты, столбцы — кислоты.
    """
    acids = list(models_nutri.keys())
    data = {acid: local_sensitivities_nutrients(acid, models_nutri, base_nutrients, step=step) for acid in acids}
    return pd.DataFrame(data, index=list(TARGET_COLS))


def build_measures_nutrients(preds: dict, sens_df: pd.DataFrame, base_nutrients: dict, top_k: int = 3, tol: float = 1e-6):
    """
    Подбор «интерпретации» для второй вкладки: какие нутриенты увеличить/уменьшить.

    В отличие от компонентного варианта, здесь нет ограничений по сумме СВ
    и «замков» — просто берём top_k нутриентов по модулю чувствительности
    с правильным знаком под требуемое направление (вверх/вниз).

    Returns:
        dict: {кислота: {'status', 'inc': [...], 'dec': [...]}}
    """
    measures = {}
    for acid, p in preds.items():
        if acid not in sens_df.columns:
            continue

        status = p['status']
        mean, lo, hi = p['mean'], p['target_min'], p['target_max']
        if '🟢' in status:
            continue

        col = sens_df[acid]
        center = 0.5 * (lo + hi)
        if mean < lo - tol:
            need_up = True
        elif mean > hi + tol:
            need_up = False
        else:
            need_up = (mean < center)

        if need_up:
            inc_candidates = col[col > 0].sort_values(ascending=False)      # что повышает кислоту
            dec_candidates = col[col < 0].abs().sort_values(ascending=False)  # что понижает — для "уменьшить"
        else:
            dec_candidates = col[col > 0].sort_values(ascending=False)      # что повышает — надо уменьшать
            inc_candidates = col[col < 0].abs().sort_values(ascending=False)  # что понижает — надо увеличивать

        inc_list = list(inc_candidates.index[:top_k])
        dec_list = list(dec_candidates.index[:top_k])
        measures[acid] = {"status": status, "inc": inc_list, "dec": dec_list}
    return measures
