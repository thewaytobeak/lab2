"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    # @dataclass(frozen=True)
    # class TitanicInput:
    #     csv_path: str

    df = pd.read_csv(data.csv_path)

    missing_by_column = df.isna().sum().to_dict()
    adults_over_30_count = int((df["Age"] > 30).sum())

    age_by_pclass = df.groupby("Pclass")["Age"].mean()
    mean_age_by_pclass = {int(k): float(v) for k, v in age_by_pclass.items()}

    survival_by_pclass = df.groupby("Pclass")["Survived"].mean()
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_by_pclass.items()}

    highest_fares = df["Fare"].nlargest(5).tolist()
    

    # @dataclass(frozen=True)
    # class TitanicSummary:
    #     row_count: int
    #     missing_by_column: dict[str, int]
    #     adults_over_30_count: int
    #     mean_age_by_pclass: dict[int, float]
    #     survival_rate_by_pclass: dict[int, float]
    #     highest_fares: list[float]


    print(len(df),
          missing_by_column,
          adults_over_30_count,
          mean_age_by_pclass,
          survival_rate_by_pclass,
          highest_fares)
    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )


if __name__ == '__main__':
    analyze_titanic(TitanicInput('titanic.csv'))