# To compile a python notebook

``` shell
jupyter nbconvert --to script your_notebook.ipynb
```



# To import config file

``` python
from ruamel.yaml import YAML

# Create YAML object
yaml = YAML()

# Open the file for reading
with open('config.yaml', 'r') as file:
    # Load the content of the file
    config = yaml.load(file)
```



