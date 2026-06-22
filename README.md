# Dragos Model Presets

A ComfyUI custom node package for managing model and LoRA presets with recommended generation settings.

## Features

- **Model Settings Node**: Load checkpoints, diffusion models, or GGUF UNet models with configurable sampler settings presets
- **LoRA Settings Node (Model+Clip)**: Apply LoRAs to both model and clip with stored strength values and metadata
- **LoRA Settings Node (Model Only)**: Apply LoRAs to model only, with `strength_clip` auto-managed in the background
- **Persistent Storage**: All presets and metadata saved in JSON files (`web/models.json` and `web/loras.json`), auto-created on first use
- **GGUF Support**: Load quantized GGUF UNet models via the `gguf_connector` package
- **Model Type Tracking**: Categorize models and LoRAs with a shared type list (Flux, SDXL, Pony, etc.)

## Installation

### Option 1: ComfyUI Manager (Recommended)

Install directly through [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) by searching for **"Dragos Model Presets"** and clicking Install.

### Option 2: Manual

Clone this repository into your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/drdrago87/Dragos-ModelPresets.git
```

Restart ComfyUI after installation.

## Node Descriptions

### Dragos Model Settings

Loads a model and provides sampler/scheduler settings that can be directly connected to a KSampler.

**Inputs:**
- `model_source_type` - Select model type: checkpoint_models, diffusion_models, or unet_models
- `model_name` - Select the model file (dynamically filtered by source type)
- `model_type` - Categorize the model (e.g., Flux .1 D, SDXL 1.0, etc.)
- `saved_setting` - Select a previously saved preset
- `turbo_lora` - Checkbox indicating if this preset requires a turbo/lightning LoRA
- `steps` - Number of sampling steps (1-1000)
- `cfg` - CFG scale (0.0-100.0, step 0.1)
- `sampler_name` - Sampling algorithm
- `scheduler` - Noise schedule
- `notes` - Freeform notes about the model

**Buttons:**
- **Save** - Updates the currently selected preset
- **Save New** - Creates a new named preset
- **Reset** - Reloads the current preset from disk
- **Delete** - Deletes the selected preset (with confirmation)

**Outputs:**
- `model` - The loaded model (MODEL type)
- `steps` - Step count (INT)
- `cfg` - CFG scale (FLOAT)
- `sampler_name` - Sampler name (STRING, compatible with KSampler)
- `scheduler` - Scheduler name (STRING, compatible with KSampler)

### Dragos LoRA Settings (Model+Clip)

Loads a LoRA and applies it to both model and clip with configurable strengths.

**Inputs:**
- `model` - Input model (MODEL)
- `clip` - Input clip (CLIP)
- `lora_name` - Select the LoRA file
- `turbo_lora` - Checkbox indicating this is a turbo/lightning LoRA
- `strength_model` - LoRA strength for model (0.00-100.00)
- `strength_clip` - LoRA strength for clip (0.00-100.00)
- `notes` - Freeform notes about the LoRA
- `lora_type` - Multi-select of compatible model types

**Buttons:**
- **Save** - Save/update the LoRA metadata
- **Reset** - Reload metadata from disk

**Outputs:**
- `model` - Modified model (MODEL)
- `clip` - Modified clip (CLIP)

### Dragos LoRA Settings (Model Only)

Same as Model+Clip but without `strength_clip`. The `strength_clip` is preserved in `loras.json` automatically.

**Inputs:**
- `model` - Input model (MODEL)
- `lora_name` - Select the LoRA file
- `turbo_lora` - Checkbox indicating this is a turbo/lightning LoRA
- `strength_model` - LoRA strength for model (0.00-100.00)
- `notes` - Freeform notes about the LoRA
- `lora_type` - Multi-select of compatible model types

**Outputs:**
- `model` - Modified model (MODEL)

## Data Files

All data files are **auto-created** when the plugin first runs — you don't need to create them manually.

### models.json

Auto-created in `web/models.json`. Stores model presets organized by source type.

```json
{
    "base_model_settings": {
        "steps": 20,
        "cfg": 8.0,
        "sampler_name": "euler",
        "scheduler": "simple"
    },
    "models": {
        "checkpoint_models": {
            "my_model.safetensors": {
                "model_type": "Flux .1 D",
                "notes": "General purpose Flux model.",
                "Default": {
                    "turbo_lora": false,
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "simple"
                },
                "Fast": {
                    "turbo_lora": true,
                    "steps": 8,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple"
                }
            }
        },
        "diffusion_models": {},
        "unet_models": {}
    }
}
```

### loras.json

Auto-created in `web/loras.json`. Stores LoRA metadata.

```json
{
    "base_lora_settings": {
        "strength_model": 1.00,
        "strength_clip": 1.00
    },
    "loras": {
        "my_lora.safetensors": {
            "lora_type": ["Flux .1 D", "Z Image Turbo"],
            "turbo_lora": false,
            "strength_model": 1.00,
            "strength_clip": 1.00,
            "notes": "General purpose style LoRA."
        }
    }
}
```

### model_types.json

Located at `web/model_types.json`. This is the shared list of model/LoRA types used by both the `model_type` dropdown (Model Settings node) and the `lora_type` multi-select (LoRA Settings nodes).

**To add a new model type**, simply edit this file and add the name to the `model_types` array. No code changes required. The API re-reads this file on every request, so changes take effect immediately without restarting ComfyUI.

```json
{
    "model_types": [
        "ACE Audio",
        "Anima",
        "...",
        "Z Image Turbo"
    ]
}
```

If this file is missing or unreadable, the plugin falls back to a minimal default list containing only `"Other"`.

## GGUF UNet Support

GGUF UNet loading is powered by the [calcuis/gguf](https://github.com/calcuis/gguf) project. The `gguf_connector` package files are included with this extension — no separate installation is needed.

Without `gguf_connector` present, the "unet_models" option will still appear but will show an error when trying to load a GGUF model. Checkpoint and diffusion model loading will work normally.

## Compatibility

- ComfyUI v0.22.1 or later
- Python 3.10+

## License

See LICENSE file.
