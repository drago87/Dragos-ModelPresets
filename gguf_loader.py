"""
GGUF Loader for Dragos Model Presets

Adapted from the calcuis/gguf ComfyUI custom node project.
Source: https://github.com/calcuis/gguf

This module provides GGUF UNet model loading support, including:
- GGMLTensor: Custom tensor subclass for quantized data
- GGMLLayer: Base module for GGML-quantized layers
- GGMLOps: Operations factory with Linear, Conv2d, Embedding, LayerNorm, GroupNorm
- GGUFModelPatcher: Model patcher for GGUF models
- load_gguf_sd: Load GGUF state dict from file
"""

import comfy.sd
import comfy.ops
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import torch
import numpy
import os
import json
import logging
import collections
import folder_paths
import inspect

# Try to import gguf_connector modules
# The user must provide the gguf_connector package from https://github.com/calcuis/gguf
GGUF_AVAILABLE = False
try:
    from .gguf_connector import reader as gr
    from .gguf_connector.quant5a import dequantize_tensor, is_quantized, is_torch_compatible
    GGUF_AVAILABLE = True
except ImportError:
    logging.warning(
        "[Dragos-ModelPresets] gguf_connector not found. "
        "GGUF UNet loading will not be available. "
        "Please copy the gguf_connector directory from https://github.com/calcuis/gguf"
    )
    # Provide fallback stubs so the rest of the code can import without error
    gr = None

    def is_quantized(tensor):
        return False

    def is_torch_compatible(tensor):
        return True

    def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(dtype) if dtype is not None else tensor
        return tensor


# ---------------------------------------------------------------------------
# Register GGUF folder paths with ComfyUI
# ---------------------------------------------------------------------------

def get_folder_names_and_paths(key, targets=[]):
    """Register GGUF-specific folder paths in ComfyUI's folder_paths."""
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = orig or base, {'.gguf'}
    if base and base != orig:
        logging.warning(f'Unknown file list already present on key {key}: {base}')


if GGUF_AVAILABLE:
    get_folder_names_and_paths('model_gguf', ['diffusion_models', 'unet'])
    get_folder_names_and_paths('clip_gguf', ['text_encoders', 'clip'])


# ---------------------------------------------------------------------------
# GGML Tensor class
# ---------------------------------------------------------------------------

class GGMLTensor(torch.Tensor):
    """Custom torch Tensor subclass that carries GGML quantization metadata."""

    def __init__(self, *args, tensor_type=None, tensor_shape=None, patches=None, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches if patches is not None else []

    def __new__(cls, *args, tensor_type=None, tensor_shape=None, patches=None, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, 'tensor_type', None)
        new.tensor_shape = getattr(self, 'tensor_shape', new.data.shape)
        new.patches = getattr(self, 'patches', []).copy()
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.debug(f"Ignoring 'copy_' on tensor: {e}")

    def empty_(self, size, *args, **kwargs):
        new_tensor = super().empty_(size, *args, **kwargs)
        return GGMLTensor(new_tensor, tensor_type=getattr(self, 'tensor_type', None),
                          tensor_shape=size, patches=getattr(self, 'patches', []).copy())

    @property
    def shape(self):
        if not hasattr(self, 'tensor_shape') or self.tensor_shape is None:
            self.tensor_shape = self.size()
        return self.tensor_shape


# ---------------------------------------------------------------------------
# Torch compiler disable helper
# ---------------------------------------------------------------------------

if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
    torch_compiler_disable = torch.compiler.disable
else:
    def torch_compiler_disable(*args, **kwargs):
        def noop(x):
            return x
        return noop


# ---------------------------------------------------------------------------
# GGML Layer base class
# ---------------------------------------------------------------------------

class GGMLLayer(torch.nn.Module):
    """Base module for GGML-quantized layers."""

    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False

    torch_compatible_tensor_types = {None}
    if GGUF_AVAILABLE:
        torch_compatible_tensor_types = {None, gr.GGMLQuantizationType.F32, gr.GGMLQuantizationType.F16}

    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f'{prefix}weight'), state_dict.get(f'{prefix}bias')
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self, torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args, **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata,
                                  strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == 'weight':
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == 'bias' and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + 'weight')
        if getattr(self.weight, 'is_largest_weight', False):
            self.largest_layer = True

    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)

    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        weight = torch.zeros_like(self.weight, device=torch.device('meta'))
        destination[prefix + 'weight'] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device('meta'))
            destination[prefix + 'bias'] = bias
        if self.largest_layer:
            shape = getattr(self.weight, 'tensor_shape', self.weight.shape)
            dtype = (torch.float16 if self.dequant_dtype == 'target'
                     or self.dequant_dtype is None else self.dequant_dtype)
            temp = torch.empty(*shape, device=torch.device('meta'), dtype=dtype)
            destination[prefix + 'temp.weight'] = temp

    def get_weight(self, tensor, dtype):
        if tensor is None:
            return None
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, 'patches', []):
            patch_list += load_patch_to_device(patches, device)
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                patch_dtype = (dtype if self.patch_dtype == 'target' else self.patch_dtype)
                weight = function(patch_list, weight, key, patch_dtype)
        return weight

    @torch_compiler_disable()
    def cast_bias_weight(self, input=None, dtype=None, device=None, bias_dtype=None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, 'dtype', torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(device)

        if self.bias is not None:
            try:
                bias = self.get_weight(self.bias.to(device), dtype)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                bias = self.get_weight(self.bias.to('cpu'), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)

        try:
            weight = self.get_weight(self.weight.to(device), dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            weight = self.get_weight(self.weight.to('cpu'), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=non_blocking, copy=False)
        return weight, bias

    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out

    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# GGMLOps: Custom operations factory for GGML layers
# ---------------------------------------------------------------------------

class GGMLOps(comfy.ops.manual_cast):
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)

    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(
                input, weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            ).to(dtype=output_dtype)

    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(
                input, self.normalized_shape, weight, bias, self.eps
            )

    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(
                input, self.num_groups, weight, bias, self.eps
            )


# ---------------------------------------------------------------------------
# GGUFModelPatcher: Model patcher for GGUF models
# ---------------------------------------------------------------------------

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    """Custom ModelPatcher that handles quantized GGUF weight patching."""

    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)
        try:
            from comfy.lora import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight
        patches = self.patches[key]

        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = load_patch_to_device(
                patches, self.load_device if self.patch_on_device else self.offload_device
            )
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', [
                    'weight', 'inplace_update'
                ])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)
            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(
                    weight, device_to, torch.float32, copy=True
                )
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype) if hasattr(comfy, 'float') and hasattr(comfy.float, 'stochastic_rounding') else out_weight.to(weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, 'patches', [])
                if len(patches) > 0:
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    mmap_released = False

    def load(self, *args, force_patch_weights=False, **kwargs):
        super().load(*args, force_patch_weights=True, **kwargs)
        if not self.mmap_released:
            linked = []
            if kwargs.get('lowvram_model_memory', 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, 'weight'):
                        device = getattr(m.weight, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, 'bias'):
                        device = getattr(m.bias, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                logging.info(f'Attempting to release mmap ({len(linked)})')
                for n, m in linked:
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        n.patch_on_device = getattr(self, 'patch_on_device', False)
        return n


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_patch_to_device(item, device):
    """Recursively move patches to the specified device."""
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(load_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [load_patch_to_device(x, device) for x in item]
    else:
        return item


def get_orig_shape(reader, tensor_name):
    """Extract the original tensor shape from GGUF metadata."""
    field_key = f'comfy.gguf.orig_shape.{tensor_name}'
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gr.GGUFValueType.ARRAY or \
       field.types[1] != gr.GGUFValueType.INT32:
        raise TypeError(
            f'Bad original shape metadata for {field_key}: '
            f'Expected ARRAY of INT32, got {field.types}'
        )
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_gguf_metadata(reader):
    """Extract all simple metadata fields from a GGUF reader."""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                if field.types[0] == gr.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gr.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gr.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gr.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except Exception:
            continue
    return metadata


# ---------------------------------------------------------------------------
# load_gguf_sd: Main GGUF state dict loader
# ---------------------------------------------------------------------------

def load_gguf_sd(path, handle_prefix='model.diffusion_model.', return_arch=False, is_extra=True):
    """
    Load a GGUF file and return a state dict suitable for ComfyUI model loading.

    Args:
        path: Path to the GGUF file
        handle_prefix: Prefix to strip from tensor names (default: 'model.diffusion_model.')
        return_arch: If True, return (state_dict, arch_string) instead of just state_dict
        is_extra: If True, return (state_dict, extra_dict) with metadata

    Returns:
        State dict with GGMLTensor values, optionally with architecture info and metadata
    """
    if not GGUF_AVAILABLE:
        raise RuntimeError(
            "GGUF loading is not available. Please install the gguf_connector package "
            "from https://github.com/calcuis/gguf"
        )

    reader = gr.GGUFReader(path)
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # Detect architecture
    compat = None
    arch_str = None

    # Try to get architecture from reader fields
    for field_name in reader.fields:
        if field_name == 'general.architecture':
            try:
                field = reader.get_field(field_name)
                if len(field.types) == 1 and field.types[0] == gr.GGUFValueType.STRING:
                    arch_str = str(field.parts[field.data[-1]], "utf-8")
            except Exception:
                pass
            break

    if arch_str is None:
        compat = 'sd.cpp'

    # Build state dict
    state_dict = {}
    qtype_dict = {}

    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        torch_tensor = torch.from_numpy(tensor.data)
        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            if compat == 'sd.cpp' and arch_str == 'sdxl':
                if any([tensor_name.endswith(x) for x in
                        ('.proj_in.weight', '.proj_out.weight')]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        if tensor.tensor_type in {gr.GGMLQuantizationType.F32, gr.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)

        state_dict[sd_key] = GGMLTensor(
            torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape
        )

        tensor_type_str = getattr(tensor.tensor_type, 'name', repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    logging.info('gguf qtypes: ' + ', '.join(f'{k} ({v})' for k, v in qtype_dict.items()))

    # Mark the largest quantized weight
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    if return_arch:
        return state_dict, arch_str

    if is_extra:
        extra = {
            "arch_str": arch_str,
            "metadata": get_gguf_metadata(reader)
        }
        return (state_dict, extra)

    return state_dict


# ---------------------------------------------------------------------------
# Public API: load_gguf_unet_model
# ---------------------------------------------------------------------------

def load_gguf_unet_model(gguf_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None, model_path=None):
    """
    Load a GGUF UNet model, compatible with ComfyUI's model loading pipeline.

    Args:
        gguf_name: Filename of the GGUF model (relative to the unet folder)
        dequant_dtype: Dequantization target dtype ('default', 'target', 'float32', etc.)
        patch_dtype: Patch dtype ('default', 'target', 'float32', etc.)
        patch_on_device: Whether to patch on device (bool)
        model_path: Optional pre-resolved full filesystem path. If provided,
            gguf_name is only used for logging. This bypasses
            folder_paths.get_full_path() so GGUF files can be loaded from
            any registered folder (diffusion_models or unet).

    Returns:
        Tuple of (model,) where model is a GGUFModelPatcher-wrapped ComfyUI model
    """
    if not GGUF_AVAILABLE:
        raise RuntimeError(
            "GGUF loading is not available. Please install the gguf_connector package "
            "from https://github.com/calcuis/gguf"
        )

    ops = GGMLOps()

    # Configure dequant_dtype
    if dequant_dtype in ('default', None):
        ops.Linear.dequant_dtype = None
    elif dequant_dtype == 'target':
        ops.Linear.dequant_dtype = dequant_dtype
    else:
        ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

    # Configure patch_dtype
    if patch_dtype in ('default', None):
        ops.Linear.patch_dtype = None
    elif patch_dtype == 'target':
        ops.Linear.patch_dtype = patch_dtype
    else:
        ops.Linear.patch_dtype = getattr(torch, patch_dtype)

    if model_path is None:
        model_path = folder_paths.get_full_path('unet', gguf_name)

    if model_path is None:
        raise FileNotFoundError(f"GGUF model not found: {gguf_name}")

    sd, extra = load_gguf_sd(model_path)

    kwargs = {}
    valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get("metadata", {})

    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}, **kwargs
    )

    if model is None:
        logging.error(f'ERROR UNSUPPORTED MODEL {model_path}')
        raise RuntimeError(f'ERROR: Could not detect model type of: {model_path}')

    model = GGUFModelPatcher.clone(model)
    model.patch_on_device = patch_on_device

    return (model,)
