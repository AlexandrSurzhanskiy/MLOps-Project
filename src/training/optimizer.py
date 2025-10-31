import torch
import logging


def get_optimizer(model, cfg):
    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {name}")

    logging.info(f"Инициализирован оптимизатор: {name.upper()} (lr={lr}, wd={weight_decay})")
    return optimizer


def get_scheduler(optimizer, cfg):
    sched_cfg = cfg.get("training", {}).get("scheduler", {})
    sched_type = sched_cfg.get("type")

    if sched_type is None:
        return None
    elif sched_type.lower() == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer,
                      step_size=sched_cfg.get("step_size", 1),
                      gamma=sched_cfg.get("gamma", 0.9))
    elif sched_type.lower() == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer,
                                 factor=sched_cfg.get("factor", 0.5),
                                 patience=sched_cfg.get("patience", 2))
    else:
        import logging
        logging.warning(f"Неизвестный тип scheduler: {sched_type}")
        return None
