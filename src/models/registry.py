import logging
from src.models.recsys_nn import RecSysNN
from src.models.baselines import ItemBasedCF, PopularityBaseline


def get_model(model_name, n_users, n_items, **kwargs):
    logging.info(f"Создание модели: {model_name}")

    if model_name == "recsys_nn":
        return RecSysNN(n_users=n_users, n_items=n_items, **kwargs)
    elif model_name == "item_cf":
        return ItemBasedCF(**kwargs)
    elif model_name == "popularity":
        return PopularityBaseline(**kwargs)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
