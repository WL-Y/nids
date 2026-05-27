import joblib

model = joblib.load("artifacts/best_model.joblib")

print("Model object:")
print(model)

print("\nModel type:")
print(type(model))

print("\nModel parameters:")
print(model.get_params())