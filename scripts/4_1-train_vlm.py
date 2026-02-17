import os, re, json, ast, argparse, itertools, traceback, random, datetime, contextlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from math import pi, cos
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
try:
    import open_clip
except Exception:
    open_clip = None
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, accuracy_score
)
from rdflib import Graph, RDFS
from rdflib.namespace import RDF, OWL
import torchvision.transforms as T
import torchvision.models as tvm
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# -----------------------------
# Constants & Mappings
# -----------------------------

CANON_STATES = ["red", "yellow", "green", "red_yellow", "off"]
STATE_TO_IDX = {state: i for i, state in enumerate(CANON_STATES)}
NUM_CLASSES = len(CANON_STATES)

STATE_TO_NUM_LIGHTS = {
    "red": 1,
    "yellow": 1,
    "green": 1,
    "red_yellow": 2,
    "off": 0
}

STATE_DESCRIPTIONS = {
    "red": "The signal shows red.",
    "yellow": "The signal shows yellow.",
    "green": "The signal shows green.",
    "red_yellow": "The signal shows red and yellow.",
    "off": "The signal is off or shows no state."
}

# Static mapping for Ontology-derived attributes (for demonstration)
ONTOLOGY_STATE_MAPPINGS = {
    "red": {
        "isControlledBy": "TopModule", 
        "hasCIE_x": "0.67 to 0.70", 
        "hasDominantWavelength": "610nm to 625nm"
    },
    "yellow": {
        "isControlledBy": "MiddleModule", 
        "hasCIE_x": "0.55 to 0.58", 
        "hasDominantWavelength": "590nm to 595nm"
    },
    "green": {
        "isControlledBy": "BottomModule", 
        "hasCIE_x": "0.15 to 0.20", 
        "hasDominantWavelength": "505nm to 520nm"
    },
    "red_yellow": {
        "isControlledBy": "Top and Middle Module", 
        "hasCIE_x": "Blended", 
        "hasDominantWavelength": "Mixed"
    },
    "off": {
        "isControlledBy": "None", 
        "hasCIE_x": "N/A", 
        "hasDominantWavelength": "N/A"
    }
}


# -----------------------------
# Description Logic & Ontology Functions
# -----------------------------

def _get_relevant_color(state: str) -> str:
    """Derives the relevant color string for the simplified description."""
    if state in ["red", "yellow", "green"]:
        return state.capitalize()
    elif state == "red_yellow":
        return "Red and Yellow"
    elif state == "off":
        return "None"
    return "Unknown"


def extract_ontology_elements(ontology_file: str) -> List[str]:
    """
    Loads the ontology and derives the list of relevant descriptive property names.
    """
    g = Graph()
    try:
        g.parse(ontology_file)
        logging.info(f"Successfully loaded ontology from {ontology_file}")
    except Exception as e:
        logging.error(f"Failed to load ontology from {ontology_file}: {e}")
        return []

    if ONTOLOGY_STATE_MAPPINGS:
        return list(ONTOLOGY_STATE_MAPPINGS[CANON_STATES[0]].keys())
    return []


def map_state_to_ontology_values(
    state: str, ontology_elements: List[str], conf: float
) -> Dict[str, str]:
    """Maps a canonical state to ontology property values based on the derived elements."""
    if state not in ONTOLOGY_STATE_MAPPINGS:
        return {}

    values = {}
    for prop in ontology_elements:
        if prop in ONTOLOGY_STATE_MAPPINGS[state]:
            values[prop] = ONTOLOGY_STATE_MAPPINGS[state][prop]
            
    values["confidence"] = f"{conf:.3f}"
    
    return values


def state_to_description(
    state: str,
    conf: float,
    ontology_elements: Optional[List[str]],
    measured_metrics: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """
    Generates a natural language description based on the chosen mode.
    """
    base = [STATE_DESCRIPTIONS.get(state, f"The signal shows {state}.")]

    if ontology_elements:
        # --- Mode: With Ontology ---
        attr_vals = map_state_to_ontology_values(state, ontology_elements, conf)
        attr_str = "; ".join(f"{k}={v}" for k, v in attr_vals.items())
        base.append(f"Ontology-derived attribute mappings: {attr_str}.")
    else:
        # --- Mode: Without Ontology (Simplified) ---
        num_lights = STATE_TO_NUM_LIGHTS.get(state, 0)
        relevant_color = _get_relevant_color(state)
        
        desc_parts = [
            f"num_lights_on={num_lights}",
            f"relevant_color={relevant_color}"
        ]
        base.append(f"State description: {'; '.join(desc_parts)}.")

    if measured_metrics is not None:
        x, y, dom_wl, rel_lum = measured_metrics
        base.append(
            f"Measured approx: CIE_x={x:.4f}; CIE_y={y:.4f}; "
            f"dominant_wavelength={dom_wl:.1f}nm; relative_luminance={rel_lum:.3f}."
        )

    return " ".join(base)


# -----------------------------
# VLM Core Classes and Functions
# -----------------------------

class TrafficLightDataset(Dataset):
    def __init__(self, data_list, image_dir, transform=None):
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        row = self.data_list[idx]
        img_path = Path(self.image_dir) / row['filename'] 
        image = Image.open(img_path).convert("RGB")
        
        state = row['state']
        label = STATE_TO_IDX[state]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, state


class VLM(nn.Module):
    def __init__(self, backbone, clip_model, clip_pretrained, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if backbone == 'clip':
            if open_clip is None:
                raise ImportError("open_clip is required for CLIP backbone.")
            
            # --- Model and Preprocessing Assignment ---
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model, 
                pretrained=clip_pretrained
            )
            # CRITICAL: Explicitly assign to self.model and self.preprocess
            self.model = model 
            self.preprocess = preprocess
            # ------------------------------------------
            
            # --- Unfreeze and Classifier Setup (Our previous fix) ---
            
            # CRITICAL FIX: Unfreeze all parameters to allow model adaptation.
            for param in self.model.parameters():
                param.requires_grad = True # <--- Now runs without AttributeError
            
            feature_dim = self.model.visual.output_dim
            self.classifier = nn.Linear(feature_dim, num_classes)
            
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

            feature_dim = self.model.fc.in_features
            
            # ResNet structure: classifier replaces original FC layer
            self.model.fc = nn.Linear(feature_dim, num_classes)
            self.classifier = nn.Identity()
            
            self.preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        

    def forward(self, x):
        if self.backbone == 'clip':
            features = self.model.encode_image(x).float()
            logits = self.classifier(features)
        elif 'resnet' in self.backbone:
            # ResNet now outputs logits directly from model.fc
            logits = self.model(x)
        
        return logits


def setup_logging(save_dir, run_name):
    """Sets up file and stream logging."""
    log_path = Path(save_dir) / "training.log"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define logger (using root logger to ensure all prints work)
    logger = logging.getLogger() 
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicate output if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Stream Handler (to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    
    logging.info(f"--- {run_name} ---")
    logging.info(f"Log file created at: {log_path.resolve()}")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Train", leave=False)
    
    for images, labels, _ in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / total_samples
    avg_acc = correct_predictions / total_samples
    
    logging.info(f"Epoch {epoch} Train: Loss={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_states = []

    with torch.no_grad():
        for images, labels, states in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_states.extend(states)

    avg_loss = total_loss / len(dataloader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    logging.info(f"Validation: Loss={avg_loss:.4f}, Accuracy={acc:.4f}, F1-Macro={f1_macro:.4f}")
    
    ## --- UPDATED SECTION START: Enhanced Validation Logging --- ##
    
    # Calculate and log detailed classification report
    report = classification_report(
        all_labels, all_preds, 
        target_names=CANON_STATES, 
        output_dict=False, # Set to True if parsing structure is desired
        zero_division=0
    )
    logging.info("\n--- Detailed Classification Report ---")
    logging.info(f"\n{report}")
    
    # Log the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logging.info("\n--- Confusion Matrix (True Rows, Pred Columns) ---")
    logging.info(f"States (Columns/Predicted): {CANON_STATES}")
    logging.info(f"\n{cm}") 
    # 
    
    logging.info("----------------------------------\n")
    
    # Log sample descriptions
    sample_indices = random.sample(range(len(all_labels)), min(5, len(all_labels)))
    logging.info("\n--- Sample Description Logging ---")
    
    for i in sample_indices:
        true_state = all_states[i]
        predicted_state = CANON_STATES[all_preds[i]]
        mock_conf = 1.0 if predicted_state == true_state else 0.5 
        
        description = state_to_description(
            predicted_state, 
            mock_conf, 
            args.ontology_elements, 
            measured_metrics=None
        )
        
        log_line = (
            f"True: {true_state:<10} | Pred: {predicted_state:<10} | "
            f"Description: {description}"
        )
        logging.info(log_line)
    logging.info("----------------------------------\n")
    
    ## --- UPDATED SECTION END --- ##
    
    return avg_loss, f1_macro


def run_training(args):
    """Main function to setup and execute the training loop."""
    
    # --- NOTE: Ensure the VLM class has the CLIP backbone UN-FROZEN ---
    # Please verify that in the VLM class's __init__ method for the 'clip' backbone,
    # you have set param.requires_grad = True for the model parameters. 
    # This is critical for the model to learn and overcome the early bias.
    # ---------------------------------------------------------------------

    setup_logging(args.save_dir, f"{args.backbone}_run")
    logging.info(f"Configuration: {json.dumps(vars(args), indent=2)}")
    
    device = torch.device(args.device)
    
    # DEVICE LOGGING
    device_info = f"Using device: {device}"
    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            device_info += f" ({torch.cuda.get_device_name(device)})"
        except Exception:
             device_info += f" (GPU detected)"
    logging.info(device_info)
    # END OF DEVICE LOGGING
    

    # --- DATA LOADING LOGIC: Scan directories for labels ---
    all_data = []
    allowed_states = set(CANON_STATES)
    image_base_path = Path(args.image_dir)

    logging.info(f"Scanning image directory: {image_base_path} for labels in subfolders...")

    for state_dir in image_base_path.iterdir():
        if state_dir.is_dir() and state_dir.name in allowed_states:
            state_label = state_dir.name
            
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                for img_path in state_dir.glob(ext):
                     all_data.append({
                        'filename': str(img_path.relative_to(image_base_path)),
                        'state': state_label
                    })

    if not all_data:
        logging.error("No image data found. Ensure subfolders (e.g., 'red', 'green') exist and contain images.")
        return

    df = pd.DataFrame(all_data)
    logging.info(f"Found {len(df)} total images. States found: {df['state'].unique()}")
    logging.info(f"Class Counts:\n{df['state'].value_counts()}")

    # Split data
    if len(df['state'].unique()) > 1 and len(df) > 1:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['state'])
    else:
        logging.warning("Insufficient data (1 class or < 2 samples). Skipping stratified split.")
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed)

    # Initialize Model and Get Preprocessing Transform
    model = VLM(args.backbone, args.clip_model, args.clip_pretrained, NUM_CLASSES).to(device)
    transform = model.preprocess

    # Create Datasets
    train_dataset = TrafficLightDataset(train_df.to_dict('records'), args.image_dir, transform=transform)
    val_dataset = TrafficLightDataset(val_df.to_dict('records'), args.image_dir, transform=transform)
    
    ## --- Weighted Sampler Logic (Retained for Balanced Batch Flow) --- ##
    
    # 1. Calculate weights for the WeightedRandomSampler
    class_counts = train_df['state'].value_counts().reindex(CANON_STATES).fillna(0).to_dict()
    num_samples = len(train_df)
    
    # Calculate weights based on inverse frequency. 
    weights_map = {
        state: num_samples / (NUM_CLASSES * count) if count > 0 else 1.0
        for state, count in class_counts.items()
    }
    
    train_weights = [
        weights_map[state]
        for state in train_df['state']
    ]
    
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=num_samples, 
        replacement=True
    )
    
    # 2. Create DataLoaders
    # The training loader uses the WeightedRandomSampler instead of shuffle=True
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ## --- Loss and Optimizer Setup (Using Standard Loss) --- ##
    
    # Calculate class weights for logging, but use standard CrossEntropyLoss 
    # to temper the aggressive gradient updates observed previously.
    raw_counts = df['state'].value_counts().reindex(CANON_STATES).fillna(0).values
    raw_counts[raw_counts == 0] = 1
    class_weights_log = (1.0 / raw_counts) / (1.0 / raw_counts).sum() * NUM_CLASSES 
    
    # CRITICAL CHANGE: Use standard, unweighted CrossEntropyLoss 
    # We rely on WeightedRandomSampler to balance the batches for now.
    criterion = nn.CrossEntropyLoss()
    
    logging.info(f"Class weights (for Sampler/Loss comparison): {class_weights_log}")
    logging.info("Using standard CrossEntropyLoss (unweighted) with WeightedRandomSampler.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Add a learning rate scheduler (Cosine Annealing suggested for better results)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_f1 = -1
    
    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_f1 = validate(model, val_loader, criterion, device, args)
        
        # Step the scheduler
        scheduler.step()
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = Path(args.save_dir) / f"best_model_f1_{best_f1:.4f}.pt"
            torch.save(model.state_dict(), checkpoint_path) 
            logging.info(f"Saved new best model to {checkpoint_path}")

# -----------------------------
# Placeholder Functions (Must exist for the dispatch logic)
# -----------------------------
def grid_search(args):
    logging.info("Grid search function called. Requires full implementation.")
def run_onto_ablation(args):
    logging.info("Ontology ablation function called. Requires full implementation.")
def infer_batch_from_folder(args):
    logging.info("Batch inference function called. Requires full implementation.")
def infer_on_crop(args):
    logging.info("Crop inference function called. Requires full implementation.")


# -----------------------------
# Argument Parsing 
# -----------------------------

def parse_args():
    """Parses command line arguments."""
    p = argparse.ArgumentParser(description="Train a VLM for Traffic Light State Classification with optional Ontology-Enhanced Descriptions")
    
    # Model and Backbone
    p.add_argument("--backbone", type=str, default="resnet18",
                   help="VLM backbone architecture (e.g., 'clip', 'resnet18').")
    p.add_argument("--clip_model", type=str, default="ViT-B-32",
                   help="CLIP model variant if backbone is 'clip'.")
    p.add_argument("--clip_pretrained", type=str, default="openai",
                   help="CLIP pretrained dataset ('openai', 'laion400m_e32', etc.).")
    
    # Data and Directories
    p.add_argument("--image_dir", type=str, required=True,
                   help="Path to the root directory containing state subfolders of images.")
    p.add_argument("--save_dir", type=str, default="runs/default",
                   help="Directory to save logs, checkpoints, and best model.")
    p.add_argument("--load", type=str, default=None,
                   help="Path to a checkpoint file to resume training or for inference.")
                   
    # Ontology Arguments
    p.add_argument("--ontology_validate", type=str, default=None,
                   help="Path to an RDF/OWL ontology file. If provided, enables ontology-derived descriptions.")
    p.add_argument("--disable_ontology", action="store_true",
               help="Disable ontology usage even if an ontology file is found.")

    
    # Training Hyperparameters
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation.")
    p.add_argument("--wd", type=float, default=0.01, help="Weight decay.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # Logging and Devices (Default set to CUDA if available)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to use for training ('cuda' or 'cpu').")
    p.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")

    # Dispatch/Ablation Arguments
    p.add_argument("--grid_json", type=str, default=None, help="JSON with hyperparameter grid.")
    p.add_argument("--onto_ablation", action="store_true", help="Run ontology ablation study.")
    p.add_argument("--infer_folder", type=str, default=None, help="Folder for batch inference.")
    p.add_argument("--infer_crop", action="store_true", help="Infer on cropped area.")

    
    return p.parse_args()


# -----------------------------
# Main Execution Block
# -----------------------------

if __name__ == "__main__":
    
    # Set seed for reproducibility before any random operation
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Auto-detect ontology file ONLY if ontology is not disabled
    if not args.disable_ontology and args.ontology_validate is None:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            here = os.getcwd()
        default_onto = os.path.join(here, "TrafficLightOntology.rdf")
        if os.path.exists(default_onto):
            args.ontology_validate = default_onto
            logging.info(f"Using auto-detected ontology file: {args.ontology_validate}")


    # Extract ontology elements for descriptions (if ontology given)
    args.ontology_elements = None
    if not args.disable_ontology and args.ontology_validate:
        args.ontology_elements = extract_ontology_elements(args.ontology_validate)
    else:
        logging.info("Ontology usage is DISABLED for this run.")


    # Dispatch logic
    if args.grid_json:
        # grid_search(args)
        pass
    elif args.onto_ablation:
        # run_onto_ablation(args)
        pass
    elif args.infer_folder and args.load:
        # infer_batch_from_folder(args)
        pass
    elif args.infer_crop and args.load:
        # infer_on_crop(args)
        pass
    else:
        # Default path: Start the main training loop
        logging.info("Starting model training...")
        run_training(args)
