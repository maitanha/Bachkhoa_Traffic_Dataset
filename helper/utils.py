from time import gmtime, strftime
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_and_expectation(
    expectations,
     predictions,
     title,
     xlabel="Index",
     ylabel="Velocity",
     save=True,
     ylim=(0,80)
     ):
    plt.figure(figsize=(35, 5))
    plt.plot(predictions, label='Predictions', marker='o', linewidth=2)
    plt.plot(expectations, label='Expectations', marker='*', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(ylim)
    # plt.xlim((0,250))
    plt.legend()
    if save:
        result_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        plt.savefig(f"./performance/predictions/prediction_{result_time}.png")
    plt.show()
    
def calculate_and_plot_difference(expectations, predictions):
    differences = np.abs(expectations - predictions)

    sorted_differences = np.sort(differences)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_differences)
    plt.title('Sorted Differences Between Expectations and Predictions')
    plt.xlabel('Data Point')
    plt.ylabel('Absolute Difference')
    plt.grid(True)
    
    plt.show()

def calculate_and_plot_grouped_differences(expectations, predictions, save=True):
    differences = np.abs(expectations - predictions)

    # Threshold values for grouping
    thresholds = [2, 4, 6, 8, 10, 12]
    threshold_step = 2

    # Initialize empty lists to store grouped differences
    grouped_differences_count = [0 for _ in range(len(thresholds))]

    # Group the differences based on the thresholds
    for diff in differences:
        for i, threshold in enumerate(thresholds):
            if diff < threshold:
                grouped_differences_count[i] += 1
                break
        else:
            # If the difference doesn't fall into any group, add it to the last group
            grouped_differences_count[-1] += 1

    plt.figure(figsize=(10, 6))
    
    group_labels = [f'<={threshold}' for threshold in thresholds]
    plt.bar(group_labels, grouped_differences_count, label='Count')
    plt.plot(grouped_differences_count, marker="o", color='red')

    plt.title('Grouped Differences Between Expectations and Predictions')
    plt.ylabel('Absolute Difference (kmh)')
    plt.grid(True)
    plt.legend()
    if save:
        result_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        plt.savefig(f"./performance/errors/error_{result_time}.png")

    plt.show()
    
def get_train_args(
    default_epochs=10, 
    default_store_model=True, 
    default_model_name="resnet_traffic_model.pth", 
    default_data_path="../data/full"
    ):
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "--epochs", 
        type=int, 
        default=default_epochs, 
        help="Number of epochs to train (default: 10)"
    )
    
    # command as python3 filename.py --store_model or python3 filename.py --no-store_model
    arg_parser.add_argument(
        "--store_model", 
        default=default_store_model,
        action=argparse.BooleanOptionalAction, 
        help="Store the trained model"
    )
    
    arg_parser.add_argument(
        "--model_name", 
        type=str, 
        default=default_model_name, 
        help="Name of the stored model file (default: 'resnet_traffic_model.pth')"
    )
    
    arg_parser.add_argument(
        "--data_path", 
        type=str, 
        default=default_data_path, 
        help="Path to the data (default: '../data/full')"
    )
    
    return arg_parser.parse_args()

def get_validation_args(
    default_model_name="resnet_traffic_model.pth", 
    default_data_path="../data/full"
    ):
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument(
        "--model_name", 
        type=str, 
        default=default_model_name, 
        help="Name of the stored model file (default: 'resnet_traffic_model.pth')"
    )
    
    arg_parser.add_argument(
        "--data_path", 
        type=str, 
        default=default_data_path, 
        help="Path to the data (default: '../data/full')"
    )
    
    return arg_parser.parse_args()