"""
Module for training the network.
"""

from src.net import DNFNet

def supervised_train(correct_net: DNFNet, train_net: DNFNet, max_epochs: int = 1000) -> tuple[list[int], list[list[int]]]:
    """
    Trains the network to convergence.
    """
    assert correct_net.shape == train_net.shape, "Networks must have the same shape"
    converged = False
    incorrect_per_epoch = []
    monomer_misses_per_epoch = []

    while not converged and len(incorrect_per_epoch) < max_epochs:
        converged = True

        max_num = 2**correct_net.input_length

        num_incorrect = 0

        monomer_misses = [0 for _ in range(train_net.num_monomers)]

        for i in range(max_num):
            # Generating the input list
            inputs = list(bin(i)[2:].zfill(correct_net.input_length))
            inputs = [1 if x == "1" else -1 for x in inputs]

            # Getting the target output and the correct activations
            target, correct_activations = correct_net(inputs = inputs)
            result, monomer_activations = train_net(inputs = inputs, target = target, train = True)

            for i in range(train_net.num_monomers):
                # Count the number of times each monomer is incorrect
                if monomer_activations[i] != correct_activations[i]:
                    monomer_misses[i] += 1

            if result != target:
                num_incorrect += 1
                converged = False

        monomer_misses_per_epoch.append(monomer_misses)
        incorrect_per_epoch.append(num_incorrect)

    monomer_misses_per_epoch = list(zip(*monomer_misses_per_epoch))
    return incorrect_per_epoch, monomer_misses_per_epoch
