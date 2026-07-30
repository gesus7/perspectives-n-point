from utils.config import load_config


def test_load_config_returns_nested_dict():
    cfg = load_config("config.yaml")
    assert "camera" in cfg
    assert "render" in cfg
    assert "train" in cfg
    assert "infer" in cfg
    assert cfg["render"]["image_size"] == [1920, 1080]
    assert cfg["model"]["num_keypoints"] == 12
