import pandas as pd
import pytest

from clustering import NonTextFieldException, NotEnoughDataException, prepare_text


# ---------------------------------------------------------------------------
# prepare_text -- joining
# ---------------------------------------------------------------------------

def test_prepare_text_joins_lists():
    data = pd.Series([["python", "pandas"], ["java", "spring"]])
    result = prepare_text(data)
    assert list(result) == ["python pandas", "java spring"]


def test_prepare_text_leaves_strings_unchanged():
    data = pd.Series(["hello world", "another title"])
    result = prepare_text(data)
    assert list(result) == ["hello world", "another title"]


def test_prepare_text_joins_a_list_that_isnt_the_first_value():
    # Regression test: list-detection used to look only at the first non-null
    # value, so a mixed field whose first row is a plain string left later
    # lists unjoined for the vectorizer to choke on.
    data = pd.Series(["already text", ["python", "pandas"]])
    result = prepare_text(data)
    assert list(result) == ["already text", "python pandas"]


def test_prepare_text_rejects_nulls_rather_than_dropping_them():
    # Dropping here would desynchronize the vectors from the DataFrame they
    # get paired with, so filtering is the caller's job (app.py does it) --
    # but a caller that forgets should get this, not a raw TypeError from
    # CountVectorizer. The message says how many rows and what to do.
    with pytest.raises(NonTextFieldException, match="1 row"):
        prepare_text(pd.Series(["python pandas", None, "java spring"]))


def test_prepare_text_raises_when_everything_is_null():
    with pytest.raises(NotEnoughDataException):
        prepare_text(pd.Series([None, None]))


# ---------------------------------------------------------------------------
# prepare_text -- validation
#
# Each of these used to reach CountVectorizer/SentenceTransformer and fail
# with a raw TypeError/ValueError, crashing the whole script run instead of
# showing app.py's clean error.
# ---------------------------------------------------------------------------

def test_prepare_text_rejects_numeric_field():
    with pytest.raises(NonTextFieldException, match="int"):
        prepare_text(pd.Series([1, 2, 3]))


def test_prepare_text_rejects_a_single_non_text_value_among_strings():
    with pytest.raises(NonTextFieldException):
        prepare_text(pd.Series(["python", "java", 7]))


def test_prepare_text_rejects_lists_holding_non_text():
    with pytest.raises(NonTextFieldException, match="list containing int"):
        prepare_text(pd.Series([["python", 3], ["java", 7]]))


def test_prepare_text_rejects_nested_objects():
    with pytest.raises(NonTextFieldException, match="dict"):
        prepare_text(pd.Series([{"a": 1}, {"b": 2}]))


def test_prepare_text_error_message_names_the_offending_type():
    with pytest.raises(NonTextFieldException) as exc:
        prepare_text(pd.Series([1.5, 2.5]))
    assert "float" in str(exc.value)
    assert "clustered as text" in str(exc.value)
