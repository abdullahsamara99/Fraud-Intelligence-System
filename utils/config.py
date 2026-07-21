import yaml


class Config:

    def __init__(self):

        with open("config/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, *keys):

        value = self.config

        for key in keys:
            value = value[key]

        return value


config = Config()