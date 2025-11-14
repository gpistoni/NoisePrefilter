from omegaconf import OmegaConf
import importlib
import torch


def initialize_training(config_path, model_ckpt_path=None):
    config = OmegaConf.load(config_path)

    config_trainer = config.trainer
    config_train_dat = config.train_dataset
    config_valid_dat = (
        config.valid_dataset if "valid_dataset" in config.keys() else None
    )

    # Instantiate the model
    model = initialize_model(config_path, model_ckpt_path)
    model = model.to("cuda")

    # Load the datasets
    train_dataset = instantiate_from_config(config_train_dat)
    valid_dataset = (
        instantiate_from_config(config_valid_dat)
        if config_valid_dat is not None
        else None
    )

    # Initialize the trainer
    print(f"Trainer Module: {config_trainer.target}")
    trainer_params = config_trainer.get("params", dict())
    trainer = get_obj_from_str(config_trainer["target"])(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        **trainer_params,
    )

    return trainer


def initialize_model(config_path, ckpt_path=None):
    # Get the config yaml and initalize the full object
    config = OmegaConf.load(config_path)
    config_model = config.model
    print(f"Target Module: {config_model.target}")
    model = instantiate_from_config(config_model)

    # If given, initialize strict from a checkpoint
    if ckpt_path is not None:
        print(f"Loading from checkpoint {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=True)
        print(
            f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

    return model


def instantiate_from_config(config_model, ckpt_path=None, strict=False):
    if not "target" in config_model:
        raise KeyError("Expected key `target` to instantiate.")
    target_str = config_model["target"]
    loaded_module = get_obj_from_str(target_str)(**config_model.get("params", dict()))

    # Get model checkpoint
    if ckpt_path is not None and ckpt_path != "None":
        print(
            f"Target: {config_model['target']} Loading from checkpoint {ckpt_path} as strict={strict}"
        )
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = loaded_module.load_state_dict(sd, strict=strict)
        print(
            f"Restored {target_str} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )

    return loaded_module


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    return getattr(importlib.import_module(module, package=None), cls)
