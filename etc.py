import yaml

def read_yaml(file: str):
    # read YAML file
    with open(file, 'r') as f:
        try:
            out = yaml.safe_load(f)
            return out
        except yaml.YAMLError as exc:
            print(exc)