from torch import nn
from ultralytics import YOLO
import os
import json

# Define constants for model weights
class Constants_Models_Weights:
    YOLOV3 = "yolov3n.pt"
    YOLOV5 = "yolov5nu.pt"
    YOLOV6 = "yolov6-n.pt"
    YOLOV8 = "yolov8n.pt"
    YOLOV9 = "yolov9c.pt"

# Define constants for model types
class Constants_Models:
    YOLOV3 = "yolov3"
    YOLOV5 = "yolov5"
    YOLOV6 = "yolov6"
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"

# Define constants for training modes
class Constants_Training:
    PRETRAINED = "pretrained"
    FROM_SCRATCH = "scratch"
    FINE_TUNING = "fine-tuning"

# Function to save data to a JSON file
def save_json(data, filename):
    if os.path.isfile(filename):
        os.remove(filename)
    json.dump(data, open(filename, 'w'), indent=4)

# Function to freeze the first 10 layers of a model during training
def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

# Define the YOLOv8 model class
class Yolov8(nn.Module):
    
    def __init__(self, args):
        super(Yolov8, self).__init__()

        # Load model based on training mode
        if args["training_mode"] == "pretrained":
            self.model = YOLO("yolov8n.pt")
            self.model.add_callback("on_train_start", freeze_layer)
        elif args["training_mode"] == "scratch":
            self.model = YOLO("./ultralytics/cfg/models/v8/yolov8.yaml")
        elif args["training_mode"] == "fine-tuning":
            self.model = YOLO("yolov8n.pt")

    def train(self, args, output_folder):
        # Train the model
        train_results = self.model.train(
            data=args["data_root"], epochs=args["epochs"], imgsz=args["imgsz"], patience=args["patience"], batch=args["batch"], 
            save=args["save"], save_period=args["save_period"], device=args["device"], workers=args["workers"], project=output_folder, 
            name=args["name"], pretrained=args["pretrained"], optimizer=args["optimizer"], seed=args["seed"], cos_lr=args["cos_lr"], 
            resume=args["resume"], lr0=args["lr0"], momentum=args["momentum"], weight_decay=args["weight_decay"], val=args["val"]
        )
        return train_results 

    def test(self, args):
        # Test the model
        test_results = self.model.val(data=args["data_root"], device=args["device"])
        return test_results

# Define the YOLOv9 model class
class Yolov9(nn.Module):
    
    def __init__(self, args):
        super(Yolov9, self).__init__()

        # Load model based on training mode
        if args["training_mode"] == "pretrained":
            self.model = YOLO("yolov9c.pt")
            self.model.add_callback("on_train_start", freeze_layer)
        elif args["training_mode"] == "scratch":
            self.model = YOLO("./ultralytics/cfg/models/v9/yolov9c_multispectral.yaml")
        elif args["training_mode"] == "fine-tuning":
            self.model = YOLO("yolov9c.pt")
        
    def train(self, args, output_folder):
        # Train the model with additional multispectral bands
        train_results = self.model.train(
            data=args["data_root"], epochs=args["epochs"], imgsz=args["imgsz"], patience=args["patience"], batch=args["batch"], 
            save=args["save"], save_period=args["save_period"], device=args["device"], workers=args["workers"], project=output_folder, 
            name=args["name"], pretrained=args["pretrained"], optimizer=args["optimizer"], seed=args["seed"], cos_lr=args["cos_lr"], 
            resume=args["resume"], lr0=args["lr0"], momentum=args["momentum"], weight_decay=args["weight_decay"], val=args["val"],
            bands_to_apply=["RGB", "Green"]  # bands to apply: list of string values
        )
        return train_results 

    def test(self, args):
        # Test the model with additional multispectral bands
        test_results = self.model.val(data=args["data_root"], device=args["device"], bands_to_apply=["RGB", "Green"])
        return test_results

# Main function to run YOLO training and testing
def yolo_v(args, output_root):
    
    # Create the output directory
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    # STEP 1: Save the training configuration in a file 
    save_json(args, os.path.join(output_root, "train_config.json"))
    
    # STEP 2: Initialize the model
    if args["model"] == Constants_Models.YOLOV9:
        model = Yolov9(args)
    elif args["model"] == Constants_Models.YOLOV8:
        model = Yolov8(args)

    # STEP 3: Train the model
    train_results = model.train(args, output_folder=output_root)
    save_json(train_results.results_dict, os.path.join(output_root, "train_results.json"))
    
    # STEP 4: Test the model on the test dataset
    test_results = model.test(args)
    save_json(test_results.results_dict, os.path.join(output_root, "test_results.json"))

    return test_results.results_dict

# Define the arguments manually
args = {
    'data_root': "D:/RESEARCH/pineapple_sam/multispectral_gira_10_13_mar21_lote71_5m/gira_10_13_mar21_lote71_5m_split/data.yaml",
    'output_root': './output/',
    'epochs': 20,
    'patience': 100,
    'batch': 16,
    'imgsz': 150,
    'save_period': -1,
    'device': "cpu",
    'workers': 4,
    'name': None,
    'pretrained': 1,
    'optimizer': 'auto',
    'seed': 0,
    'resume': 0,
    'lr0': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'model': 'yolov9',
    'model_weights': 'yolov9c.pt',
    'numa': -1,
    'training_mode': "fine-tuning",  # pretrained, scratch, fine-tuning
    'save': True,
    'cos_lr': True, 
    'val': False
}

# Run the YOLO training and testing
yolo_v(args, './output/')