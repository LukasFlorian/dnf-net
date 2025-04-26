"""
Helper functions for the project.
"""

from random import random
import matplotlib.pyplot as plt

def random_adjustment(factor: float = 1) -> float:
    """
    Returns a random value between -1 and 1
    """
    return random()*(-1 if random() < 0.5 else 1)*factor


def plot_network_results(network_misses: list[int|float]) -> None:
    """
    Plots the number of network misses per epoch.
    """
    plt.xlabel('Epoch number')
    plt.ylabel('Network misses')
    plt.title('Incorrect network outputs per epoch')

    plt.plot(network_misses)

    _, top = plt.ylim()
    plt.ylim(bottom = 0, top = top)

    plt.show()

def plot_monomer_results(monomer_misses_per_epoch: list[list[int|float]]) -> None:
    """
    Plots the number of monomer misses per epoch.
    """
    plt.xlabel("Epoch")
    plt.ylabel("Monomer misses")
    plt.title("Number of monomer misses per epoch")
    
    
    num_monomers = len(monomer_misses_per_epoch)
    for i in range(num_monomers):
        plt.plot(monomer_misses_per_epoch[i], label=f"Monomer {i+1}")
    
    _, top = plt.ylim()
    plt.ylim(bottom = 0, top = top)
    
    plt.legend()
    plt.show()

def sma(series: list[int], window: int) -> list[float]:
    """
    Calculates the simple moving average of a series.
    """
    return [sum(series[i:i+window])/window for i in range(len(series)-window+1)]

def visualize_results(network_misses: list[int], monomer_misses_per_epoch: list[list[int]], window: int = 100) -> None:
    """
    Visualizes the results of the training.
    """
    
    plot_network_results(network_misses)

    plot_monomer_results(monomer_misses_per_epoch)

    plot_combined_sma(network_misses, monomer_misses_per_epoch, window)
    

def plot_combined_sma(incorrect_per_epoch: list[int], monomer_misses_per_epoch: list[list[int]], window: int) -> None:
    """
    Plots the SMA of the network's and each monomer's number of misses per epoch.
    """
    plt.xlabel("Epoch")
    plt.ylabel(f"SMA-{window} of the Monomer and network misses")
    plt.title("Number of monomer and network misses per epoch")
    
    
    num_epochs = len(incorrect_per_epoch)
    
    x_axis = [i+window for i in range(num_epochs-window+1)]
    
    network_sma = sma(incorrect_per_epoch, 100)
    plt.plot(x_axis, network_sma, label="Network")

    for i, monomer_misses in enumerate(monomer_misses_per_epoch):
        monomer_sma = sma(monomer_misses, 100)
        plt.plot(x_axis, monomer_sma, label=f"Monomer {i+1}")
    
    _, top = plt.ylim()
    plt.ylim(bottom = 0, top = top)
    
    plt.legend()
    plt.show()
    
def plot_several_networks(network_misses: list[list[int]], names: list[str], window: int = 1) -> None:
    """
    Plots the number of misses per epoch for several networks.
    """
    if window > 1:
        for i, misses in enumerate(network_misses):
            network_misses[i] = sma(misses, window)
        plt.title(f"SMA-{window} of the number of misses per epoch at different learning rates")
    else:
        plt.title("Number of misses per epoch for different learning rates")
    
    plt.xlabel("Epoch")
    plt.ylabel("Number of misses")

    for i, misses in enumerate(network_misses):
        plt.plot(misses, label=f"{names[i]}")

    _, top = plt.ylim()
    plt.ylim(bottom = 0, top = top)

    plt.legend()
    plt.show()