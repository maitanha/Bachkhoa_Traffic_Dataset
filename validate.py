from model import Model
from operator import attrgetter
from helper.utils import plot_prediction_and_expectation, get_validation_args,calculate_and_plot_grouped_differences

if __name__ == "__main__":
    args = get_validation_args(
        default_model_name="resnet_velocity_full_v2.pth", 
        default_data_path="./data/full"
    )
    
    data_path, model_name = attrgetter(
        'data_path', 
        'model_name')(args)
    
    validation_model = Model(data_path)
    expectations, predictions = validation_model.validate(model_name)
    
    plot_prediction_and_expectation(
        predictions=predictions, 
        expectations=expectations, 
        title="Expectation vs Prediction"
    )
    
    calculate_and_plot_grouped_differences(
        predictions=predictions,
        expectations=expectations
    )
    
    print("Result images are saved in ./performance")