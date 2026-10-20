"""Shared primitives used by both clustering backends (clustering/lexical.py,
clustering/semantic.py) that aren't specific to KMeans or either backend's
own vectorization/scoring approach.
"""
import pandas as pd


class NotEnoughDataException(Exception):
    pass


class NonTextFieldException(Exception):
    """The chosen field holds values neither backend can vectorize as text
    (numbers, nested objects, lists of them, missing values, ...).

    Raised deliberately instead of coercing with str(): the app's
    preprocessing options are all opt-in precisely so nothing is transformed
    without the user seeing it, and clustering numbers as if they were words
    produces a confident, meaningless answer -- worse than no answer. Without
    this, CountVectorizer/SentenceTransformer fail with a raw TypeError that
    crashes the whole script run instead of app.py's clean error.
    """


def _describe_bad_value(value) -> str:
    if isinstance(value, list):
        inner = next((v for v in value if not isinstance(v, str)), None)
        return f"a list containing {type(inner).__name__}"
    return type(value).__name__


def prepare_text(data: pd.Series) -> pd.Series:
    """Validate that a field is text (or lists of text) and join any lists
    into one string per row -- the single entry point both backends use to
    turn a raw JSON field into vectorizable text.

    List-valued fields are detected from the data's actual type rather than
    configured, so a `tags` array needs no separate handling by the caller.
    Nulls are rejected rather than dropped: dropping rows here would
    desynchronize the vectors from the DataFrame app.py pairs them with, so
    filtering is the caller's job (app.py's drop_missing_field_values does it
    before this point). Refusing loudly keeps a future caller that forgets
    from getting a raw TypeError out of CountVectorizer instead.
    """
    non_null = data.dropna()
    if non_null.empty:
        raise NotEnoughDataException
    if len(non_null) < len(data):
        raise NonTextFieldException(
            f"This field has no value on {len(data) - len(non_null)} row(s); "
            "filter those rows out before vectorizing."
        )

    bad = next(
        (
            v for v in non_null
            if not isinstance(v, (str, list))
            or (isinstance(v, list) and any(not isinstance(item, str) for item in v))
        ),
        None,
    )
    if bad is not None:
        raise NonTextFieldException(
            f"This field contains {_describe_bad_value(bad)}, which can't be clustered as text."
        )

    # Any list anywhere means the field is list-valued -- checking only the
    # first non-null value would leave a later list to reach the vectorizer
    # unjoined.
    if not any(isinstance(v, list) for v in non_null):
        return data
    return data.apply(lambda v: " ".join(v) if isinstance(v, list) else v)
