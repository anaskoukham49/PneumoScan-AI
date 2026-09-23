import cv2
import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        raise ValueError("Could not find a Conv2D layer in the model.")

    with tf.GradientTape() as tape:
        x = img_array
        last_conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                last_conv_output = x
                tape.watch(last_conv_output)
        
        preds = x
        if pred_index is None:
            pred_index = 0
        class_channel = preds[:, pred_index]

    if last_conv_output is None:
        raise ValueError(f"Layer {last_conv_layer_name} not found.")

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0)
    max_heat = tf.math.reduce_max(heatmap)
    if max_heat != 0:
        heatmap = heatmap / max_heat
    heatmap = heatmap.numpy()
        
    return heatmap

def overlay_gradcam(original_image_bytes, heatmap, alpha=0.4):
    """
    Overlays the heatmap onto the original image.
    original_image_bytes: original image bytes.
    heatmap: 2D numpy array with values 0 to 1.
    Returns: BGR numpy array image.
    """
    np_buffer = np.frombuffer(original_image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file for Grad-CAM overlay.")
    
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img
