"""
Example usage of the DNFNet class, the supervised_train function and the visualize_results function.
"""

from src.net import DNFNet
from src.train import supervised_train
from src.helpers import visualize_results

# Referenznetzwerk mit vorberechneten Gewichten erstellen
correct_model = DNFNet(input_length=10, num_monomers=5)


# Parameter entsprechend der DNF-Formel setzen:
correct_model.monomer_weights = [
    [1, 1, 1, -1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, -1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, -1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, -1, -1, -1]
]

correct_model.monomer_biases = [4, 4, 4, 4, 4]

correct_model.output_weights = [1, 1, 1, 1, 1]

correct_model.output_bias = -3



# Zu trainierendes Netzwerk erstellen
train_net = DNFNet(input_length=10, num_monomers=5, learning_rate=0.1)

# Netzwerk trainieren
incorrect_per_epoch, monomer_misses = supervised_train(
    correct_model,
    train_net,
    max_epochs=1000
)

# Ergebnisse visualisieren
visualize_results(incorrect_per_epoch, monomer_misses)
