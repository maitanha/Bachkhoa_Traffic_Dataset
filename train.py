from model import Model
from helper.utils import get_train_args
from operator import attrgetter

if __name__ == "__main__":
    args = get_train_args(
        default_epochs=10, 
        default_store_model=True, 
        default_model_name="resnet_traffic_model.pth", 
        default_data_path="./data/full"
    )
    
    data_path, epochs, store_model, model_name = attrgetter(
        'data_path', 
        'epochs', 
        'store_model', 
        'model_name')(args)
    
    train_model = Model(data_path)
    
    # Train the model with 10 epochs and save the model as "resnet_traffic_model.pth"
    train_model.train(
        epochs=epochs, 
        store_model=store_model, 
        model_name=model_name
    )