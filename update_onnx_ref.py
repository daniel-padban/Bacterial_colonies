import onnx


# Load the ONNX model
model = onnx.load("models/bacterial_plate_model_v1.onnx")

# Convert external data references
onnx.load_external_data_for_model(model, "models/bacterial_plate_model_v1.onnx.data")

# Save the updated ONNX model
onnx.save(model, "models/bacterial_plate_model_v1.onnx")