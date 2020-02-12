import src.utils.helper_functions as hf

def test_date_time():
    date = fn.get_time_string(True)
    assert len(date) == 18
    date = fn.get_time_string(False)
    assert len(date) == 16