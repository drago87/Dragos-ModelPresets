from .nodes import DragosModelSettingsNode, DragosLoraSettingsNodeModelClip, DragosLoraSettingsNodeModelOnly

NODE_CLASS_MAPPINGS = {
    "DragosModelSettings": DragosModelSettingsNode,
    "DragosLoraSettingsModelClip": DragosLoraSettingsNodeModelClip,
    "DragosLoraSettingsModelOnly": DragosLoraSettingsNodeModelOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DragosModelSettings": "Dragos Model Settings",
    "DragosLoraSettingsModelClip": "Dragos LoRA Settings (Model+Clip)",
    "DragosLoraSettingsModelOnly": "Dragos LoRA Settings (Model Only)",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
