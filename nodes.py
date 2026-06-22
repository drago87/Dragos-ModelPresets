"""
Dragos Model Presets - ComfyUI Custom Nodes

Provides three nodes for managing model and LoRA presets:
1. DragosModelSettingsNode - Load models with configurable sampler settings
2. DragosLoraSettingsNodeModelClip - Apply LoRA to model and clip
3. DragosLoraSettingsNodeModelOnly - Apply LoRA to model only

Data is persisted in web/models.json and web/loras.json.
Custom API routes handle save/load/delete operations from the frontend.

Model index architecture:
- A unified MODEL_INDEX is built at startup by scanning both
  diffusion_models and unet folders using os.walk(), bypassing
  folder_paths.get_filename_list() entirely to avoid any weird
  registration behavior from ComfyUI or third-party GGUF extensions.
- diffusion_models dropdown: .safetensors files from either folder
- gguf_models dropdown: .gguf files from either folder
- Duplicate detection prevents the same file appearing twice
- find_model_path() resolves a filename from either folder by
  checking the actual directory structure
"""

import json
import os
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import folder_paths
import comfy.sd
import comfy.utils
import comfy.samplers
import nodes

from server import PromptServer
from aiohttp import web

from .gguf_loader import GGUF_AVAILABLE, load_gguf_unet_model


# ===========================================================================
# Constants
# ===========================================================================

PKG_DIR = Path(__file__).parent
MODELS_JSON_PATH = PKG_DIR / "web" / "models.json"
LORAS_JSON_PATH = PKG_DIR / "web" / "loras.json"
MODEL_TYPES_JSON_PATH = PKG_DIR / "web" / "model_types.json"

# Thread lock for JSON file writes
_json_lock = threading.Lock()

# Base settings defaults
BASE_MODEL_SETTINGS = {
    "steps": 20,
    "cfg": 8.0,
    "sampler_name": "euler",
    "scheduler": "simple"
}

BASE_LORA_SETTINGS = {
    "strength_model": 1.00,
    "strength_clip": 1.00
}

# Default fallback list — the actual list is loaded from web/model_types.json
_DEFAULT_MODEL_TYPES = ["Other"]


def _load_model_types() -> list:
    """Load the model type list from web/model_types.json.

    Returns the list of model type strings. If the file is missing or
    unreadable, falls back to the hardcoded default.
    """
    try:
        if MODEL_TYPES_JSON_PATH.exists():
            with open(MODEL_TYPES_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            types = data.get("model_types", [])
            if types:
                return types
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"[Dragos-ModelPresets] Error reading model_types.json: {e}. Using defaults.")
    return _DEFAULT_MODEL_TYPES


MODEL_TYPE_LIST = _load_model_types()

# Model source type options — gguf_models replaces the old unet_models
MODEL_SOURCE_TYPES = ["checkpoint_models", "diffusion_models", "gguf_models"]

# Sampler and scheduler lists from ComfyUI
def get_sampler_names():
    return comfy.samplers.KSampler.SAMPLERS


def get_scheduler_names():
    return comfy.samplers.KSampler.SCHEDULERS


# ===========================================================================
# Unified Model Index
# ===========================================================================

# Cached index built at startup and on ComfyUI refresh.
# Maps source_type -> sorted list of full ComfyUI filenames.
MODEL_INDEX = {
    "diffusion_models": [],
    "gguf_models": []
}


def rebuild_model_index():
    """
    Build a unified model index.

    diffusion_models:
        All .safetensors found in:
            models/diffusion_models
            models/unet

    gguf_models:
        All .gguf found in:
            models/diffusion_models
            models/unet

    Duplicate detection is based on relative path.
    """
    global MODEL_INDEX

    safetensors = []
    ggufs = []

    models_root = folder_paths.models_dir

    folders_to_scan = [
        os.path.join(models_root, "diffusion_models"),
        os.path.join(models_root, "unet"),
    ]

    # rel_path -> first folder it was found in
    seen = {}

    for folder in folders_to_scan:
        if not os.path.isdir(folder):
            continue

        folder_name = os.path.basename(folder)

        for root, _dirs, files in os.walk(folder):
            for file in files:
                lower = file.lower()

                if not (
                    lower.endswith(".safetensors")
                    or lower.endswith(".gguf")
                ):
                    continue

                full_path = os.path.join(root, file)

                rel_path = os.path.relpath(
                    full_path,
                    folder
                ).replace("\\", "/")

                rel_key = rel_path.lower()

                if rel_key in seen:
                    logging.warning(
                        "[Dragos-ModelPresets] Duplicate found: "
                        f"{folder_name}/{rel_path} and "
                        f"{seen[rel_key]}/{rel_path} "
                        "- keeping first"
                    )
                    continue

                seen[rel_key] = folder_name

                if lower.endswith(".safetensors"):
                    safetensors.append(rel_path)
                else:
                    ggufs.append(rel_path)

    safetensors.sort(key=str.lower)
    ggufs.sort(key=str.lower)

    MODEL_INDEX = {
        "diffusion_models": safetensors,
        "gguf_models": ggufs,
    }

    logging.info(
        f"[Dragos-ModelPresets] Indexed "
        f"{len(safetensors)} safetensors and "
        f"{len(ggufs)} gguf models"
    )


def find_model_path(model_name: str):
    """Resolve a model filename to its full filesystem path.

    Searches both diffusion_models and unet folders by checking the actual
    directory structure, bypassing ComfyUI's registration system entirely.

    model_name is a relative path like 'Flux/my_model.gguf'. This function
    combines it with each registered folder root and checks if the file exists.
    """
    # Normalize separators
    normalized = model_name.replace("\\", "/")

    for folder_type in ("diffusion_models", "unet"):
        try:
            folders = folder_paths.get_folder_paths(folder_type)
        except Exception:
            continue
        for folder in folders:
            candidate = os.path.join(folder, normalized)
            if os.path.isfile(candidate):
                return candidate

    return None


# Build the index at import time (startup)
rebuild_model_index()


# ===========================================================================
# JSON Data Management
# ===========================================================================

def _ensure_json_file(path: Path, default_data: dict) -> dict:
    """Ensure a JSON file exists and return its contents. Create with defaults if missing."""
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"[Dragos-ModelPresets] Error reading {path}: {e}. Recreating.")
    # Create the file with default data
    with _json_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default_data, f, indent=4, ensure_ascii=False)
    return default_data.copy()


def _read_json(path: Path) -> dict:
    """Read and return JSON data from a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"[Dragos-ModelPresets] Error reading {path}: {e}")
        return {}


def _write_json(path: Path, data: dict) -> bool:
    """Write data to a JSON file with thread safety."""
    with _json_lock:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            logging.error(f"[Dragos-ModelPresets] Error writing {path}: {e}")
            return False


def _get_default_models_data() -> dict:
    """Return the default models.json structure."""
    return {
        "base_model_settings": BASE_MODEL_SETTINGS.copy(),
        "models": {
            "checkpoint_models": {},
            "diffusion_models": {},
            "gguf_models": {}
        }
    }


def _get_default_loras_data() -> dict:
    """Return the default loras.json structure."""
    return {
        "base_lora_settings": BASE_LORA_SETTINGS.copy(),
        "loras": {}
    }


def get_models_data() -> dict:
    """Get models.json data, creating the file if it doesn't exist."""
    return _ensure_json_file(MODELS_JSON_PATH, _get_default_models_data())


def get_loras_data() -> dict:
    """Get loras.json data, creating the file if it doesn't exist."""
    return _ensure_json_file(LORAS_JSON_PATH, _get_default_loras_data())


# ===========================================================================
# Name Utilities
# ===========================================================================

def _clean_name(name: str) -> str:
    """Strip folder path and file extension from a model/LoRA filename.

    Used to convert full ComfyUI filenames into clean JSON storage keys.

    Converts e.g. 'Flux/my_model.gguf' to 'my_model'.
    Converts e.g. 'Characters/Anime/style.safetensors' to 'style'.
    """
    # Handle both / and \ separators
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    # Strip the file extension (last . and everything after)
    if "." in basename:
        basename = basename.rsplit(".", 1)[0]
    return basename


# ===========================================================================
# Model / LoRA List Helpers
# ===========================================================================

def get_model_list(source_type: str) -> list:
    """Get model filenames for a given source type.

    - checkpoint_models: queried directly from folder_paths
    - diffusion_models / gguf_models: served from the unified MODEL_INDEX
    """
    if source_type == "checkpoint_models":
        try:
            return folder_paths.get_filename_list("checkpoints")
        except Exception as e:
            logging.warning(f"[Dragos-ModelPresets] Error getting checkpoints: {e}")
            return []
    return MODEL_INDEX.get(source_type, [])


def get_lora_list() -> list:
    """Get LoRA filenames from ComfyUI."""
    try:
        return folder_paths.get_filename_list("loras")
    except Exception as e:
        logging.warning(f"[Dragos-ModelPresets] Error getting lora list: {e}")
        return []


def get_model_settings(source_type: str, model_name: str) -> dict:
    """
    Get all settings for a specific model from models.json.

    The model_name from the dropdown is a full ComfyUI filename (e.g.
    'Flux/my_model.gguf'). It is converted to a clean name for JSON
    lookup (e.g. 'my_model').
    """
    clean_key = _clean_name(model_name)
    data = get_models_data()
    source_models = data.get("models", {}).get(source_type, {})

    # Try the clean key first (standard case)
    model_data = source_models.get(clean_key, {})

    # Fallback: try the raw model_name (backward compatibility with old
    # data that may have been saved under the full name)
    if not model_data:
        model_data = source_models.get(model_name, {})

    result = {
        "model_type": model_data.get("model_type", ""),
        "notes": model_data.get("notes", ""),
        "settings": {}
    }

    # Extract setting entries (everything except model_type and notes)
    for key, value in model_data.items():
        if key not in ("model_type", "notes"):
            result["settings"][key] = value

    return result


def get_lora_metadata(lora_name: str) -> dict:
    """Get metadata for a specific LoRA from loras.json.

    The lora_name from the dropdown is a full ComfyUI filename.
    It is converted to a clean name for JSON lookup.
    """
    clean_key = _clean_name(lora_name)
    data = get_loras_data()
    loras_dict = data.get("loras", {})

    # Try the clean key first (standard case)
    lora_data = loras_dict.get(clean_key, {})

    # Fallback: try the raw lora_name (backward compatibility)
    if not lora_data:
        lora_data = loras_dict.get(lora_name, {})

    if not lora_data:
        # Return defaults
        return {
            "lora_type": [],
            "turbo_lora": False,
            "strength_model": data.get("base_lora_settings", BASE_LORA_SETTINGS).get("strength_model", 1.00),
            "strength_clip": data.get("base_lora_settings", BASE_LORA_SETTINGS).get("strength_clip", 1.00),
            "notes": ""
        }
    return lora_data


# ===========================================================================
# API Routes
# ===========================================================================

routes = PromptServer.instance.routes


@routes.get('/dragos/data/models')
async def api_get_models_data(request):
    """Return the full models.json data."""
    data = get_models_data()
    return web.json_response(data)


@routes.get('/dragos/data/loras')
async def api_get_loras_data(request):
    """Return the full loras.json data."""
    data = get_loras_data()
    return web.json_response(data)


@routes.get('/dragos/model_list')
async def api_get_model_list(request):
    """Return the list of model filenames for a given source type."""
    source_type = request.rel_url.query.get("source_type", "checkpoint_models")
    model_list = get_model_list(source_type)
    return web.json_response({"models": model_list})


@routes.get('/dragos/lora_list')
async def api_get_lora_list(request):
    """Return the list of LoRA filenames."""
    lora_list = get_lora_list()
    return web.json_response({"loras": lora_list})


@routes.get('/dragos/model_types')
async def api_get_model_types(request):
    """Return the model type list from web/model_types.json (re-reads each time so edits are picked up without restart)."""
    return web.json_response({"model_types": _load_model_types()})


@routes.get('/dragos/samplers')
async def api_get_samplers(request):
    """Return the available sampler and scheduler names."""
    return web.json_response({
        "sampler_names": list(comfy.samplers.KSampler.SAMPLERS),
        "scheduler_names": list(comfy.samplers.KSampler.SCHEDULERS),
    })


@routes.get('/dragos/model_settings')
async def api_get_model_settings(request):
    """Return settings for a specific model."""
    source_type = request.rel_url.query.get("source_type", "")
    model_name = request.rel_url.query.get("model_name", "")
    if not source_type or not model_name:
        return web.json_response({"error": "Missing source_type or model_name"}, status=400)
    settings = get_model_settings(source_type, model_name)
    return web.json_response(settings)


@routes.get('/dragos/lora_metadata')
async def api_get_lora_metadata(request):
    """Return metadata for a specific LoRA."""
    lora_name = request.rel_url.query.get("lora_name", "")
    if not lora_name:
        return web.json_response({"error": "Missing lora_name"}, status=400)
    metadata = get_lora_metadata(lora_name)
    return web.json_response(metadata)


@routes.post('/dragos/rebuild_model_index')
async def rebuild_model_index_api(request):
    """Rebuild the unified model index (called on ComfyUI refresh)."""
    rebuild_model_index()
    return web.json_response({"success": True})


@routes.post('/dragos/models/save')
async def api_save_model_setting(request):
    """Save/update a model setting."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    source_type = data.get("source_type", "")
    model_name = _clean_name(data.get("model_name", ""))
    setting_name = data.get("setting_name", "")
    model_type = data.get("model_type", "")
    turbo_lora = data.get("turbo_lora", False)
    steps = data.get("steps", 20)
    cfg = data.get("cfg", 8.0)
    sampler_name = data.get("sampler_name", "euler")
    scheduler = data.get("scheduler", "simple")
    notes = data.get("notes", "")

    if not source_type or not model_name or not setting_name:
        return web.json_response({"error": "Missing required fields"}, status=400)

    # Validate steps and cfg
    steps = max(1, min(1000, int(steps)))
    cfg = max(0.0, min(100.0, float(cfg)))

    models_data = get_models_data()

    # Ensure the source type and model entry exist
    if source_type not in models_data.get("models", {}):
        return web.json_response({"error": f"Invalid source type: {source_type}"}, status=400)

    if model_name not in models_data["models"][source_type]:
        models_data["models"][source_type][model_name] = {
            "model_type": model_type,
            "notes": notes
        }
    else:
        # Update model-level metadata
        if model_type:
            models_data["models"][source_type][model_name]["model_type"] = model_type
        if notes is not None:
            models_data["models"][source_type][model_name]["notes"] = notes

    # Save/update the setting
    models_data["models"][source_type][model_name][setting_name] = {
        "turbo_lora": bool(turbo_lora),
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler_name,
        "scheduler": scheduler
    }

    if _write_json(MODELS_JSON_PATH, models_data):
        return web.json_response({"status": "ok"})
    else:
        return web.json_response({"error": "Failed to write models.json"}, status=500)


@routes.post('/dragos/models/save_new')
async def api_save_new_model_setting(request):
    """Create a new named setting for a model."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    source_type = data.get("source_type", "")
    model_name = _clean_name(data.get("model_name", ""))
    new_setting_name = data.get("new_setting_name", "")

    if not source_type or not model_name or not new_setting_name:
        return web.json_response({"error": "Missing required fields"}, status=400)

    models_data = get_models_data()

    # Check if setting name already exists
    model_entry = models_data.get("models", {}).get(source_type, {}).get(model_name, {})
    if new_setting_name in model_entry:
        return web.json_response({"error": f"Setting '{new_setting_name}' already exists"}, status=409)

    # Get base settings for defaults
    base = models_data.get("base_model_settings", BASE_MODEL_SETTINGS)

    # Create the new setting with provided values or defaults
    new_setting = {
        "turbo_lora": data.get("turbo_lora", False),
        "steps": max(1, min(1000, int(data.get("steps", base["steps"])))),
        "cfg": max(0.0, min(100.0, float(data.get("cfg", base["cfg"])))),
        "sampler_name": data.get("sampler_name", base["sampler_name"]),
        "scheduler": data.get("scheduler", base["scheduler"])
    }

    # Ensure model entry exists
    if source_type not in models_data.get("models", {}):
        return web.json_response({"error": f"Invalid source type: {source_type}"}, status=400)

    if model_name not in models_data["models"][source_type]:
        models_data["models"][source_type][model_name] = {
            "model_type": data.get("model_type", ""),
            "notes": data.get("notes", "")
        }

    models_data["models"][source_type][model_name][new_setting_name] = new_setting

    if _write_json(MODELS_JSON_PATH, models_data):
        return web.json_response({"status": "ok", "setting_name": new_setting_name})
    else:
        return web.json_response({"error": "Failed to write models.json"}, status=500)


@routes.post('/dragos/models/delete')
async def api_delete_model_setting(request):
    """Delete a specific setting from a model."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    source_type = data.get("source_type", "")
    model_name = _clean_name(data.get("model_name", ""))
    setting_name = data.get("setting_name", "")

    if not source_type or not model_name or not setting_name:
        return web.json_response({"error": "Missing required fields"}, status=400)

    models_data = get_models_data()
    model_entry = models_data.get("models", {}).get(source_type, {}).get(model_name, {})

    if setting_name not in model_entry:
        return web.json_response({"error": f"Setting '{setting_name}' not found"}, status=404)

    del models_data["models"][source_type][model_name][setting_name]

    # Keep model entry with model_type and notes even if no settings remain
    if _write_json(MODELS_JSON_PATH, models_data):
        return web.json_response({"status": "ok"})
    else:
        return web.json_response({"error": "Failed to write models.json"}, status=500)


@routes.post('/dragos/loras/save')
async def api_save_lora(request):
    """Save/update a LoRA entry."""
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    lora_name = _clean_name(data.get("lora_name", ""))
    if not lora_name:
        return web.json_response({"error": "Missing lora_name"}, status=400)

    loras_data = get_loras_data()
    base = loras_data.get("base_lora_settings", BASE_LORA_SETTINGS)

    # Check if this is a new entry or an update
    is_new = lora_name not in loras_data.get("loras", {})

    lora_entry = {
        "lora_type": data.get("lora_type", []),
        "turbo_lora": bool(data.get("turbo_lora", False)),
        "strength_model": max(0.0, float(data.get("strength_model", base["strength_model"]))),
        "notes": data.get("notes", "")
    }

    if is_new:
        # New entry: add strength_clip from base settings
        lora_entry["strength_clip"] = float(data.get("strength_clip", base["strength_clip"]))
    else:
        # Update: preserve strength_clip if not explicitly provided
        existing_clip = loras_data["loras"][lora_name].get("strength_clip", base["strength_clip"])
        lora_entry["strength_clip"] = float(data.get("strength_clip", existing_clip))

    # Round strength values to 2 decimal places
    lora_entry["strength_model"] = round(lora_entry["strength_model"], 2)
    lora_entry["strength_clip"] = round(lora_entry["strength_clip"], 2)

    loras_data.setdefault("loras", {})[lora_name] = lora_entry

    if _write_json(LORAS_JSON_PATH, loras_data):
        return web.json_response({"status": "ok"})
    else:
        return web.json_response({"error": "Failed to write loras.json"}, status=500)


# ===========================================================================
# Node Classes
# ===========================================================================

class DragosModelSettingsNode:
    """
    Dragos Model Settings Node

    Loads a model (checkpoint, diffusion model, or GGUF UNet) and applies
    saved preset configuration. Outputs the model along with steps, cfg,
    sampler_name, and scheduler values that can be connected to a KSampler.
    """

    CATEGORY = "Dragos/Model Presets"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source_type": (MODEL_SOURCE_TYPES, {"default": "checkpoint_models"}),
    
                # Dynamic widget managed entirely by JS
                "model_name": ("STRING", {"default": ""}),
    
                "model_type": (MODEL_TYPE_LIST, {"default": "Other"}),
                "saved_setting": (["(none)"],),
                "turbo_lora": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (get_sampler_names(),),
                "scheduler": (get_scheduler_names(),),
                "notes": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL", "INT", "FLOAT", "*", "*")
    RETURN_NAMES = ("model", "steps", "cfg", "sampler_name", "scheduler")
    FUNCTION = "process"
    OUTPUT_NODE = False

    def process(self, model_source_type, model_name, model_type, saved_setting,
                turbo_lora, steps, cfg, sampler_name, scheduler, notes, node_id=None):
        """Load the model and return it with the current settings."""
        # Clamp values
        steps = max(1, min(1000, int(steps)))
        cfg = max(0.0, min(100.0, float(cfg)))

        # Load the model based on source type
        model = self._load_model(model_source_type, model_name)

        return (model, steps, cfg, sampler_name, scheduler)

    def _load_model(self, source_type, model_name):
        """Load a model from the specified source.

        model_name is a full ComfyUI filename (e.g. 'Flux/my_model.gguf')
        which can be passed directly to folder_paths.get_full_path() or
        find_model_path().
        """
        if not model_name or model_name == "(no models found)":
            raise ValueError("No model selected or no models available.")

        if source_type == "checkpoint_models":
            ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
            if ckpt_path is None:
                raise FileNotFoundError(f"Checkpoint not found: {model_name}")
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            return out[0]  # MODEL

        elif source_type == "diffusion_models":
            model_path = find_model_path(model_name)
            if model_path is None:
                raise FileNotFoundError(f"Diffusion model not found: {model_name}")
            return comfy.sd.load_diffusion_model(model_path)

        elif source_type == "gguf_models":
            if not GGUF_AVAILABLE:
                raise RuntimeError(
                    "GGUF loading is not available. Please install the gguf_connector "
                    "package from https://github.com/calcuis/gguf"
                )
            model_path = find_model_path(model_name)
            if model_path is None:
                raise FileNotFoundError(f"GGUF model not found: {model_name}")
            return load_gguf_unet_model(model_name, model_path=model_path)[0]

        else:
            raise ValueError(f"Unknown model source type: {source_type}")


class DragosLoraSettingsNodeModelClip:
    """
    Dragos LoRA Settings Node (Model + Clip)

    Loads a LoRA and applies stored strengths and metadata to
    both the model and clip inputs.
    """

    CATEGORY = "Dragos/Model Presets"

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = get_lora_list()

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (lora_list if lora_list else ["(no loras found)"],),
                "turbo_lora": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "strength_model": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.01}),
                "notes": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "process"
    OUTPUT_NODE = False

    def process(self, model, clip, lora_name, turbo_lora, strength_model,
                strength_clip, notes, node_id=None):
        """Load and apply the selected LoRA to model and clip."""
        if not lora_name or lora_name == "(no loras found)":
            raise ValueError("No LoRA selected or no LoRAs available.")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora_data, strength_model, strength_clip
        )

        return (model_lora, clip_lora)


class DragosLoraSettingsNodeModelOnly:
    """
    Dragos LoRA Settings Node (Model Only)

    Loads a LoRA and applies stored strength to the model input only.
    strength_clip is preserved in loras.json but not exposed in this node.
    """

    CATEGORY = "Dragos/Model Presets"

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = get_lora_list()

        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (lora_list if lora_list else ["(no loras found)"],),
                "turbo_lora": ("BOOLEAN", {"default": False, "label_on": "Yes", "label_off": "No"}),
                "strength_model": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 100.0, "step": 0.01}),
                "notes": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "process"
    OUTPUT_NODE = False

    def process(self, model, lora_name, turbo_lora, strength_model, notes, node_id=None):
        """Load and apply the selected LoRA to model only."""
        if not lora_name or lora_name == "(no loras found)":
            raise ValueError("No LoRA selected or no LoRAs available.")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise FileNotFoundError(f"LoRA not found: {lora_name}")

        lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, _ = comfy.sd.load_lora_for_models(
            model, None, lora_data, strength_model, 0.0
        )

        return (model_lora,)
