import yaml

# f = open("configuration.yml")
# file = yaml.safe_load(f)
# f.close()

with open("configuration.yml", "r") as stream:
    config = yaml.safe_load(stream)

print(config)

list = config["learning_rate"][0]
print(list)

param = config["neuron_units"]["Dense0"]
print(param)