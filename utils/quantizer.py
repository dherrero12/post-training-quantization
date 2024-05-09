# Import required packages
import torch
import numpy as np

# Quantize classification models
def quantize_models(models, model_args):
    # Return unquantized classification models
    if model_args.quantization is None:
        return models

    # Quantize classification models
    quantized_models = {}
    quantized_sizes = {} 
    model_sizes = {}
    for task_name, model in models.items():
        if model_args.quantization == 'absmax':
            model, quant_size, model_size = quantize_model(model, lambda X: absmax_quantize(X, model_args.bits))
        elif model_args.quantization == 'zeropoint':
            model, quant_size, model_size = quantize_model(model, lambda X: zeropoint_quantize(X, model_args.bits))
        elif model_args.quantization == 'norm':
            model, quant_size, model_size = quantize_model(model, lambda X: norm_quantize(X, model_args.quantile))
        elif model_args.quantization == 'partial':
            model, quant_size, model_size = quantize_model(model, lambda X: partial_quantize(X, model_args.quantile, model_args.bits))
        else:
            raise ValueError('Unknown quantization scheme')
        
        # Store quantized model and sizes
        quantized_models[task_name] = model
        quantized_sizes[task_name] = quant_size
        model_sizes[task_name] = model_size

    # Return quantized classification models
    return quantized_models, quantized_sizes, model_sizes

# Absmax quantize tensor 
def absmax_quantize(X, bits):
    # Check valid number of bits
    assert isinstance(bits, int)

    # Calculate scale on tensor
    half_max = 2 ** (bits - 1) - 1
    scale = half_max  / torch.max(torch.abs(X))

    # Quantize tensor
    X_quant = (scale * X).round()
    X_dequant = X_quant / scale
    X_dequant = X_dequant.type(X.dtype)

    # Compute parameter size
    num_elements = X_quant.nelement()
    quant_size = num_elements * (bits / 8)
    dequant_size = num_elements * X_quant.element_size()

    # Return dequantized tensor
    return X_dequant, quant_size, dequant_size

# Zero-point quantize tensor
def zeropoint_quantize(X, bits):
    # Compute value range (denominator)
    x_range = torch.max(X) - torch.min(X)
    x_range = 1 if x_range == 0 else x_range

    # Compute scale
    max_ = 2 ** bits - 1
    half_max = (max_ + 1) // 2
    scale = max_ / x_range

    # Shift by zero-point
    zeropoint = (-scale * torch.min(X) - half_max).round()

    # Scale and round the inputs
    X_quant = torch.clip((X * scale + zeropoint).round(), -half_max, half_max-1)

    # Dequantize tensor
    X_dequant = (X_quant - zeropoint) / scale
    X_dequant = X_dequant.type(X.dtype)

    # Compute parameter size
    num_elements = X_quant.nelement()
    quant_size = num_elements * (bits / 8)
    dequant_size = num_elements * X_quant.element_size()

    # Return dequantized tensor
    return X_dequant, quant_size, dequant_size

# Norm threshold tensor
def norm_quantize(X, quantile):
    # Check valid quantile
    assert isinstance(quantile, float) and 0 <= quantile <= 1
    
    # Compute threshold mask
    X_abs = X.abs()
    threshold = np.quantile(X_abs, quantile)
    mask = X.abs() >= threshold

    # Mask under threshold values
    X_thresh = X * mask
    X_thresh = X_thresh.type(X.dtype)

    # Compute parameter size
    index_bits = np.ceil(np.log2(np.max(X.shape)))
    quant_size = (X_thresh.ndim * (index_bits / 8) + X_thresh.element_size()) * float(mask.sum())
    model_size = X_thresh.nelement() * X_thresh.element_size()

    # Return thresholded tensor
    return X_thresh, quant_size, model_size

def partial_quantize(X, quantile, bits):
    # Check valid quantile and bits
    assert isinstance(quantile, float) and 0 <= quantile <= 1
    assert isinstance(bits, int)

    # Define threshold based on quantile
    X_abs = X.abs()
    threshold = np.quantile(X_abs, quantile)

    # Define masks for the quantization
    mask_above = X_abs >= threshold
    mask_below = ~mask_above

    # Quantize weights below the threshold
    X_below = X * mask_below
    X_quant_below, _, _ = absmax_quantize(X_below, bits)
    X_quant_below = torch.nan_to_num(X_quant_below, 0)

    # Combine quantized and original weights
    X_quant = X_quant_below + X * mask_above

    # Compute parameter size 
    index_bits = np.ceil(np.log2(np.max(X.shape)))
    num_elements = X.nelement()
    quant_size = num_elements * (bits / 8) + (X.ndim * (index_bits / 8) + X.element_size()) * float(mask_above.sum())
    model_size = num_elements * X_quant.element_size()

    # Return quantized tensor
    return X_quant, quant_size, model_size

# Quantize parameters of model
def quantize_model(model, quantizer):
    # Initialize model sizes
    quant_size = model_size = 0

    # Quantize parameters of model 
    for param in model.parameters():
        quantized_param, quant_size_param, model_size_param = quantizer(param.data)
        param.data = quantized_param
        quant_size += quant_size_param
        model_size += model_size_param

    # Quantize buffers of model
    for buffer_ in model.buffers():
        quantized_buffer, quant_size_buffer, model_size_buffer = quantizer(buffer_.data)
        buffer_.data = quantized_buffer
        quant_size += quant_size_buffer
        model_size += model_size_buffer

    # Convert model sizes into MB
    quant_size = quant_size / (1024 ** 2)
    model_size = model_size / (1024 ** 2)
    
    # Return quantized model
    return model, quant_size, model_size
