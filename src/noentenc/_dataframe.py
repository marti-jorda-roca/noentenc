from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


def column_values(dataset: pl.DataFrame | pd.DataFrame, column: str) -> list[Any]:
    _frame_module(dataset)
    return dataset[column].to_list()


def with_column(
    dataset: pl.DataFrame | pd.DataFrame,
    column: str,
    values: list[Any],
    *,
    strings: bool = False,
) -> pl.DataFrame | pd.DataFrame:
    """Return `dataset` with `column` set to `values`.

    `strings` types a polars column as String even when every value is null.
    """
    if _frame_module(dataset) == "polars":
        # The caller passed a polars frame, so polars is installed.
        import polars

        frame = cast("pl.DataFrame", dataset)
        return frame.with_columns(
            polars.Series(column, values, polars.String if strings else None)
        )
    return cast("pd.DataFrame", dataset).assign(**{column: values})


def _frame_module(dataset: object) -> str:
    module = type(dataset).__module__.split(".", 1)[0]
    if module not in {"polars", "pandas"}:
        raise TypeError(
            f"expected a polars or pandas DataFrame, got {type(dataset).__name__}"
        )
    return module
