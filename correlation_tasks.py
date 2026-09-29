"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput
FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

def mri_correlation(mri_data: pd.DataFrame) -> dict[str, float]:
    return {f: float(mri_data[f].corr(mri_data["MRI_Count"])) for f in FEATURES}

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA", "?"])

    women = df[df["Gender"] == "Female"]
    men = df[df["Gender"] == "Male"]

    women_mri_correlation = mri_correlation(women)
    men_mri_correlation = mri_correlation(men)

    pairs = [
        (abs(corr), feature)
        for corr_map in (women_mri_correlation, men_mri_correlation)
        for feature, corr in corr_map.items()
    ]
    
    strongest_mri_feature = max(pairs)[1]

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature,
    )