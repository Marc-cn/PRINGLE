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

from rdflib import URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD

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

# New Constants for Method 2: Attribute Encoding
# ---------------------------------------------------------------------
# 1. Maps each unique attribute value to an integer index.
ATTRIBUTE_VALUE_MAPS = {
    "isControlledBy": {
        "TopModule": 0,
        "MiddleModule": 1,
        "BottomModule": 2,
        "Top and Middle Module": 3,
        "None": 4,
    },
    # For CIE_x and Wavelength, we treat them as predicting the 'bin'
    "hasCIE_x": {
        "0.67 to 0.70": 0,  # Red range
        "0.55 to 0.58": 1,  # Yellow range
        "0.15 to 0.20": 2,  # Green range
        "Blended": 3,
        "N/A": 4,
    },
    "hasDominantWavelength": {
        "610nm to 625nm": 0,  # Red range
        "590nm to 595nm": 1,  # Yellow range
        "505nm to 520nm": 2,  # Green range
        "Mixed": 3,
        "N/A": 4,
    }
}

# The number of attributes we are predicting (length of the outer dictionary)
NUM_ONTOLOGY_ELEMENTS = len(ATTRIBUTE_VALUE_MAPS)
# ---------------------------------------------------------------------


# -----------------------------
# Description Logic & Ontology Functions
# -----------------------------

def _lit_to_str(v):
    if isinstance(v, Literal):
        return str(v)
    return str(v)

def _find_state_node(g: Graph, state: str) -> Optional[URIRef]:
    """
    Find the RDF resource representing a given state string (e.g., 'red').
    Strategy (robust):
      1) look for rdfs:label == state (case-insensitive)
      2) fallback: subject URI endswith '/red' or '#red'
    """
    target = state.strip().lower()

    # 1) rdfs:label match
    for s, _, label in g.triples((None, RDFS.label, None)):
        if isinstance(label, Literal) and str(label).strip().lower() == target:
            return s

    # 2) URI suffix match
    for s in g.subjects(None, None):
        if isinstance(s, URIRef):
            ss = str(s).lower()
            if ss.endswith(f"#{target}") or ss.endswith(f"/{target}") or ss.endswith(target):
                return s

    return None

def extract_ontology_elements(ontology_file: str) -> List[str]:
    """
    Load ontology and return a list of properties we want to use as attributes.
    Minimal option: keep your existing 3 attributes as an allow-list, but confirm
    they exist in the ontology. (This avoids schema guessing.)
    """
    g = Graph()
    try:
        g.parse(ontology_file)
        logging.info(f"Successfully loaded ontology from {ontology_file}")
    except Exception as e:
        logging.error(f"Failed to load ontology from {ontology_file}: {e}")
        return []

    allow = ["isControlledBy", "hasCIE_x", "hasDominantWavelength"]

    # Keep only properties that appear at least once in the graph
    present = set()
    for _, p, _ in g:
        if isinstance(p, URIRef):
            local = p.split("#")[-1] if "#" in p else p.split("/")[-1]
            if local in allow:
                present.add(local)

    elems = [a for a in allow if a in present]
    logging.info(f"Ontology elements enabled from RDF: {elems}")
    return elems

def build_attribute_value_maps_from_ontology(g: Graph, ontology_elements: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Build ATTRIBUTE_VALUE_MAPS dynamically: for each property, gather all distinct
    values in the ontology and map them to integer ids.
    """
    maps: Dict[str, Dict[str, int]] = {}
    for elem in ontology_elements:
        values = set()
        for _, p, o in g.triples((None, None, None)):
            if isinstance(p, URIRef):
                local = p.split("#")[-1] if "#" in p else p.split("/")[-1]
                if local != elem:
                    continue
                values.add(_lit_to_str(o))

        # Stable ordering
        values_sorted = sorted(values)
        maps[elem] = {v: i for i, v in enumerate(values_sorted)}

    return maps

def map_state_to_ontology_values_from_graph(
    g: Graph, state: str, ontology_elements: List[str], conf: float
) -> Dict[str, str]:
    """
    Query RDF to retrieve property values for the given state node.
    Returns {prop_local_name: value_as_string, ..., "confidence": "..."}.
    """
    node = _find_state_node(g, state)
    if node is None:
        return {"confidence": f"{conf:.3f}"}

    out: Dict[str, str] = {}
    for _, p, o in g.triples((node, None, None)):
        if not isinstance(p, URIRef):
            continue
        local = p.split("#")[-1] if "#" in p else p.split("/")[-1]
        if local in ontology_elements:
            out[local] = _lit_to_str(o)

    out["confidence"] = f"{conf:.3f}"
    return out


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
        # Return the keys that are actually used as attributes
        return list(ATTRIBUTE_VALUE_MAPS.keys())
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
    measured_metrics: Optional[Tuple[float, float, float]] = None, ontology_values: Optional[Dict[str,str]]=None) -> str:
    """
    Generates a natural language description based on the chosen mode.
    """
    base = [STATE_DESCRIPTIONS.get(state, f"The signal shows {state}.")]

    if ontology_elements:
        # --- Mode: With Ontology (Enhanced for Method 3 Prompt) ---
        # Note: We now use ontology_elements=list(ATTRIBUTE_VALUE_MAPS.keys())
        attr_vals = ontology_values if ontology_values is not None else map_state_to_ontology_values(state, ontology_elements, conf)
        
        # Format the rich prompt text
        attr_str_list = [f"{k} is {v}" for k, v in attr_vals.items() if k != 'confidence']
        attr_str = ", ".join(attr_str_list)
        
        # Base sentence + Ontology details (for CLIP prompt)
        base = [f"A photo of a traffic light showing the {state} state, where the signal attributes are: {attr_str}."]
        
    else:
        # --- Mode: Without Ontology (Simplified) ---
        num_lights = STATE_TO_NUM_LIGHTS.get(state, 0)
        relevant_color = _get_relevant_color(state)
        
        desc_parts = [
            f"num_lights_on={num_lights}",
            f"relevant_color={relevant_color}"
        ]
        base.append(f"State description: {'; '.join(desc_parts)}.")

    if measured_metrics is not None and len(measured_metrics) >= 3:
        x, y, dom_wl = measured_metrics[0], measured_metrics[1], measured_metrics[2]
        base.append(
            # Log the predicted values here:
            f"Predicted approx: CIE_x={x:.4f}; CIE_y={y:.4f}; "
            f"dominant_wavelength={dom_wl:.1f}nm."
        )

    return " ".join(base)


# -----------------------------
# VLM Core Classes and Functions
# -----------------------------

class TrafficLightDataset(Dataset):
    """
    Dataset for traffic light images, providing targets for multi-task learning:
    1. Primary Classification (label)
    2. Attribute Classification (attr_targets)
    3. Contrastive Learning (ontology_text)
    4. Continuous Regression (regression_targets)
    """
    
    # --- UPDATED: Accept and store regression_stats ---
    def __init__(self, data_list, image_dir, transform=None, ontology_elements=None, regression_stats=None, ontology_graph=None):
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform = transform
        self.ontology_elements = ontology_elements 
        self.regression_stats = regression_stats # Store stats for normalization
        self.ontology_graph = ontology_graph
        
    # --- INCLUDED: __len__ function ---
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

        # Query real ontology values (if graph exists)
        attr_values = {}
        if self.ontology_graph is not None and self.ontology_elements:
            attr_values = map_state_to_ontology_values_from_graph(
                self.ontology_graph, state, self.ontology_elements, conf=1.0
            )
            
        # 1. Target for Method 3 (Contrastive Loss)
        # Generate the rich, ontology-enhanced text description
        ontology_text = state_to_description(
            state, 
            conf=1.0, 
            ontology_elements=self.ontology_elements, 
            measured_metrics=None,
            ontology_values=attr_values
        )

        # 2. Target for Method 2 (Attribute Loss)
        # Create numerical targets based on the state's ontological values
        attr_targets = []
        if self.ontology_elements and ATTRIBUTE_VALUE_MAPS:
            for prop in self.ontology_elements:
                value = attr_values.get(prop, "N/A")
                target_idx = ATTRIBUTE_VALUE_MAPS.get(prop, {}).get(value, 0)
                attr_targets.append(target_idx)
            attr_targets = torch.tensor(attr_targets, dtype=torch.long)
        else:
             # Return empty tensor if ontology is disabled/not available
            attr_targets = torch.zeros(NUM_ONTOLOGY_ELEMENTS, dtype=torch.long)

        # 3. Continuous Regression Targets (Load and Normalize) 
        
        # Load actual values from 'row' (using .get() for safety)
        # Assuming your dataframe rows contain 'gt_cie_x', 'gt_cie_y', 'gt_dom_wl'
        regression_targets = torch.tensor([
            row.get('gt_cie_x', 0.0),        # Index 0
            row.get('gt_cie_y', 0.0),        # Index 1
            row.get('gt_dom_wl', 0.0)      # Index 2
        ], dtype=torch.float)

        # Apply Normalization (Z-score: (Value - Mean) / StdDev)
        if self.regression_stats:
            stats = self.regression_stats
            
            # CIE_x (Index 0)
            if stats['cie_x']['std'] > 1e-6:
                regression_targets[0] = (regression_targets[0] - stats['cie_x']['mean']) / stats['cie_x']['std']
            
            # CIE_y (Index 1)
            if stats['cie_y']['std'] > 1e-6:
                regression_targets[1] = (regression_targets[1] - stats['cie_y']['mean']) / stats['cie_y']['std']
            
            # Dom_WL (Index 2)
            if stats['dom_wl']['std'] > 1e-6:
                regression_targets[2] = (regression_targets[2] - stats['dom_wl']['mean']) / stats['dom_wl']['std']

        # Return image, primary label, state string, ontology text, attribute targets, and regression targets
        return image, label, state, ontology_text, attr_targets, regression_targets


class VLM(nn.Module):
    def __init__(self, backbone, clip_model, clip_pretrained, num_classes, num_ontology_elements): # <-- num_ontology_elements is passed for convenience/logging
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        # Use the actual number of predicted attributes (length of ATTRIBUTE_VALUE_MAPS)
        self.num_ontology_elements = len(ATTRIBUTE_VALUE_MAPS) 

        if backbone == 'clip':
            if open_clip is None:
                raise ImportError("open_clip is required for CLIP backbone.")
            
            # --- Model and Preprocessing Assignment ---
            model, _, preprocess = open_clip.create_model_and_transforms(
                clip_model, 
                pretrained=clip_pretrained
            )
            self.model = model 
            self.preprocess = preprocess
            
            # CRITICAL FIX: Unfreeze all parameters to allow model adaptation.
            for param in self.model.parameters():
                param.requires_grad = True 
            
            feature_dim = self.model.visual.output_dim
            self.classifier = nn.Linear(feature_dim, num_classes)
            
            # NEW for Method 2: Auxiliary Head for Ontology Attributes
            if self.num_ontology_elements > 0:
                # Calculate total output size needed (sum of unique values for all attributes)
                total_attr_outputs = sum(len(v) for v in ATTRIBUTE_VALUE_MAPS.values())
                # Single linear head to predict logits for all attributes combined
                self.aux_head = nn.Linear(feature_dim, total_attr_outputs) 
            else:
                self.aux_head = nn.Identity()
            
            # NEW for Method 3: Text Tokenizer
            self.tokenizer = open_clip.get_tokenizer(clip_model)

            # --- NEW for Continuous Regression (Method 4) ---
            self.NUM_REGRESSION_TARGETS = 3 # For (CIE_x, CIE_y, Dominant Wavelength)
            self.regression_head = nn.Linear(feature_dim, self.NUM_REGRESSION_TARGETS)
            
        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

            feature_dim = self.model.fc.in_features
            
            self.model.fc = nn.Linear(feature_dim, num_classes)
            self.classifier = nn.Identity()
            
            self.preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.tokenizer = None # No tokenizer for ResNet
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        

    def forward(self, x, ontology_text_list=None):
        if self.backbone == 'clip':
            # 1. Image Encoding (Visual Features)
            visual_features = self.model.encode_image(x).float()
            
            # 2. Text Encoding (Semantic Features) - Method 3
            text_features = None
            if ontology_text_list is not None and self.tokenizer is not None:
                # Tokenize the list of ontology strings
                text_tokens = self.tokenizer(ontology_text_list).to(visual_features.device)
                text_features = self.model.encode_text(text_tokens).float()
            
            # 3. State Prediction (Primary Task)
            logits = self.classifier(visual_features)
            
            # 4. Attribute Prediction (Auxiliary Task) - Method 2
            aux_logits = self.aux_head(visual_features)

            # 5. Continuous Regression Prediction (Method 4)
            regression_output = self.regression_head(visual_features)
            
            return logits, aux_logits, regression_output, visual_features, text_features # <--- New return value
            
        elif 'resnet' in self.backbone:
            # ResNet: Standard forward pass
            logits = self.model(x)
            
            # Return placeholder features/logits for ResNet (no multi-task/text)
            return logits, None, None, None 


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
    total_samples = 0
    total_correct = 0
    
    # Define weights for the multi-task loss (Hyperparameters to tune)
    AUX_LOSS_WEIGHT = 0.095     # Weight for Method 2 (Attribute Loss)
    CONTRASTIVE_LOSS_WEIGHT = 0.523 # Weight for Method 3 (Contrastive Loss)
    REGRESSION_LOSS_WEIGHT = 0.380 # <--- New hyperparameter

    # Use L1 or MSE Loss for regression
    regression_criterion = nn.MSELoss()
    
    # Determine the start/end indices for the auxiliary prediction heads
    attr_start_indices = {}
    current_index = 0
    attr_names = list(ATTRIBUTE_VALUE_MAPS.keys())
    for attr_name, val_map in ATTRIBUTE_VALUE_MAPS.items():
        attr_start_indices[attr_name] = current_index
        current_index += len(val_map)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Train", leave=False)
    
    # Dataloader loop receives two extra returns
    for images, labels, _, ontology_texts, attr_targets, regression_targets in progress_bar: 
        
        # Move inputs to device
        images, labels = images.to(device), labels.to(device)
        if attr_targets is not None:
             attr_targets = attr_targets.to(device)
        regression_targets = regression_targets.to(device)
        optimizer.zero_grad()
        
        # Model call with text input
        logits, aux_logits, regression_output, visual_features, text_features = model(images, ontology_texts)        
        # --- Loss Component 1: State Classification (Primary Task) ---
        primary_loss = criterion(logits, labels)
        loss = primary_loss
        
        # --- Loss Component 2: Attribute Prediction (Method 2) ---
        attribute_loss = torch.tensor(0.0, device=device)
        if aux_logits is not None and model.num_ontology_elements > 0:
            
            # The auxiliary head output (aux_logits) is (Batch_Size x Total_Attribute_Outputs)
            # We slice it to get the logits for each individual attribute prediction.
            
            for i, attr_name in enumerate(attr_names):
                start_idx = attr_start_indices[attr_name]
                num_values = len(ATTRIBUTE_VALUE_MAPS[attr_name])
                
                # Slice the logits for the current attribute: (Batch_Size x Num_Values_in_Attribute)
                attr_logits = aux_logits[:, start_idx : start_idx + num_values]
                
                # The target for the current attribute is attr_targets[:, i]: (Batch_Size)
                attr_target = attr_targets[:, i]
                
                # Calculate CrossEntropyLoss for the current attribute prediction
                attribute_loss += criterion(attr_logits, attr_target)
            
            # Average the loss across the number of predicted attributes
            attribute_loss /= model.num_ontology_elements

            loss += AUX_LOSS_WEIGHT * attribute_loss

        # --- NEW Loss Component 4: Regression Loss ---
        regression_loss = regression_criterion(regression_output, regression_targets)
        loss += REGRESSION_LOSS_WEIGHT * regression_loss
        
        # --- Loss Component 3: Contrastive Loss (Method 3) ---
        contrastive_loss = torch.tensor(0.0, device=device)
        if visual_features is not None and text_features is not None:
            # Normalize features
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Use cosine distance loss (1 - cosine similarity)
            contrastive_loss = torch.mean(1.0 - torch.sum(visual_features * text_features, dim=-1))
            
            loss += CONTRASTIVE_LOSS_WEIGHT * contrastive_loss
          
        # ------------------- End Loss Calculation -------------------
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        # Use primary logits for progress bar tracking
        _, predicted = torch.max(logits, 1) 
        total_samples += labels.size(0)

        total_correct += (predicted == labels).sum().item()
        
        # Log all loss components in the progress bar
        progress_bar.set_postfix({
            'L_Comb': f'{loss.item():.4f}',
            'L_P': f'{primary_loss.item():.4f}',
            'L_A': f'{attribute_loss.item():.4f}',
            'L_C': f'{contrastive_loss.item():.4f}'
        })

    avg_loss = total_loss / total_samples
    # We don't track per-batch accuracy here, rely on validate for accurate metric
    # avg_acc = 0.0 
    avg_acc = total_correct / total_samples
    
    logging.info(f"Epoch {epoch} Train: Loss (Combined)={avg_loss:.4f}, Accuracy={avg_acc:.4f}")
    return avg_loss, avg_acc

def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_states = []
    # NEW: Store all regression predictions
    all_regression_preds = []

    with torch.no_grad():
        for images, labels, states, _, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # --- FIX 1: Capture Regression Output ---
            if model.backbone == 'clip':
                # Model returns: logits, aux_logits, regression_output, visual_features, text_features
                logits, _, regression_output, _, _ = model(images, ontology_text_list=None) 
            else:
                logits, _, _, _ = model(images)
                regression_output = None # Placeholder if not clip
                
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_states.extend(states)

            # NEW: Store predictions (Convert from (B, 3) tensor to list of tuples)
            if regression_output is not None:
                all_regression_preds.extend(regression_output.cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    logging.info(f"Epoch Validate: Loss={avg_loss:.4f}, Accuracy={acc:.4f}, Macro F1={f1_macro:.4f}")
    
    sample_indices = random.sample(range(len(all_labels)), min(5, len(all_labels)))
    logging.info("\n--- Sample Description Logging (Pred. Continuous Values) ---")
    
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
        
        # --- FIX 2: Get predicted continuous metrics ---
        predicted_metrics = None
        if all_regression_preds:
            # Get the predicted CIE_x, CIE_y, and Dom_WL for this sample
            predicted_metrics = tuple(all_regression_preds[i])

        # Pass the predicted metrics to state_to_description
        description = state_to_description(
            predicted_state, 
            mock_conf, 
            args.ontology_elements, 
            measured_metrics=predicted_metrics # <-- Now passes the predicted continuous values
        )
        
        log_line = (
            f"True: {true_state:<10} | Pred: {predicted_state:<10} | "
            f"Description: {description}"
        )
        logging.info(log_line)
    logging.info("----------------------------------\n")
    
    return avg_loss, f1_macro


def run_training(args):
    """Main function to setup and execute the training loop."""
    
    setup_logging(args.save_dir, f"{args.backbone}_run")
    logging.info(f"Configuration: {json.dumps(vars(args), indent=2)}")

    ontology_graph = None
    if args.ontology_elements and args.ontology_validate:
        ontology_graph = Graph()
        ontology_graph.parse(args.ontology_validate)
        # Build dynamic maps
        maps = build_attribute_value_maps_from_ontology(ontology_graph, args.ontology_elements)

        # ✅ FIX: remove attributes that have no values in RDF
        maps = {k: v for k, v in maps.items() if len(v) > 0}

        # Keep args consistent with what we can actually supervise
        args.ontology_elements = list(maps.keys())

        # Build dynamic ATTRIBUTE_VALUE_MAPS from RDF
        global ATTRIBUTE_VALUE_MAPS, NUM_ONTOLOGY_ELEMENTS
        ATTRIBUTE_VALUE_MAPS = build_attribute_value_maps_from_ontology(ontology_graph, args.ontology_elements)
        NUM_ONTOLOGY_ELEMENTS = len(ATTRIBUTE_VALUE_MAPS)

        logging.info(f"Dynamic ATTRIBUTE_VALUE_MAPS built from ontology: {json.dumps(ATTRIBUTE_VALUE_MAPS, indent=2)}")
        logging.info(f"Ontology elements used for training (filtered): {args.ontology_elements}")
    
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

    # --- NEW: Calculate Normalization Stats on Training Data ---
    REGRESSION_STATS = None
    if all(col in train_df.columns for col in ['gt_cie_x', 'gt_cie_y', 'gt_dom_wl']):
        REGRESSION_STATS = {
            'cie_x': {'mean': train_df['gt_cie_x'].mean(), 'std': train_df['gt_cie_x'].std()},
            'cie_y': {'mean': train_df['gt_cie_y'].mean(), 'std': train_df['gt_cie_y'].std()},
            'dom_wl': {'mean': train_df['gt_dom_wl'].mean(), 'std': train_df['gt_dom_wl'].std()}
        }
        logging.info(f"Regression Stats (Mean/Std): {json.dumps({k: {k2: f'{v2:.4f}' for k2, v2 in v.items()} for k, v in REGRESSION_STATS.items()}, indent=2)}")
    else:
        logging.warning("Skipping regression target normalization: Regression ground truth columns (gt_cie_x, etc.) not found in DataFrame.")

    # Initialize Model and Get Preprocessing Transform
    # ---------------------------------------------------------------------
    # Determine the number of ontology elements to predict
    num_onto_elements = len(args.ontology_elements) if args.ontology_elements else 0
    
    # Pass num_onto_elements to VLM
    model = VLM(args.backbone, args.clip_model, args.clip_pretrained, NUM_CLASSES, num_onto_elements).to(device)
    transform = model.preprocess

    # Create Datasets
    # Pass ontology_elements to TrafficLightDataset
    train_dataset = TrafficLightDataset(train_df.to_dict('records'), args.image_dir, transform=transform, 
                                        ontology_elements=args.ontology_elements, regression_stats=REGRESSION_STATS, ontology_graph=ontology_graph)
    val_dataset = TrafficLightDataset(val_df.to_dict('records'), args.image_dir, transform=transform, 
                                      ontology_elements=args.ontology_elements, regression_stats=REGRESSION_STATS, ontology_graph=ontology_graph)
    # ---------------------------------------------------------------------
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
