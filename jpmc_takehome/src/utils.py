from typing import Union


def get_classifier_params(params: dict[str, Union[int, float, str]]):
    return {
        p.removeprefix("cls__"): v for p, v in params.items() if p.startswith("cls__")
    }


def get_select_params(params: dict[str, Union[int, float, str]]):
    return {
        p.removeprefix("select__"): v
        for p, v in params.items()
        if p.startswith("select__")
    }
