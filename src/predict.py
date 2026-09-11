import joblib
import pandas as pd

def derive_condition(light_lux, noise_db):
    # Same thresholds used when the training data was generated
    is_bright = light_lux > 700
    is_noisy = noise_db > 50
    if is_bright and is_noisy:
        return "Both"
    elif is_bright:
        return "Light"
    elif is_noisy:
        return "Noise"
    else:
        return "Baseline"

def predict_reaction_time(light_lux, noise_db, n_back_level,
                           age=24, gender="Male", trial_number=13,
                           model_path="models/best_model_random_forest.pkl"):
    model = joblib.load(model_path)
    condition = derive_condition(light_lux, noise_db)

    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "condition": condition,
        "light_lux": light_lux,
        "noise_db": noise_db,
        "n_back_level": n_back_level,
        "trial_number": trial_number
    }])

    prediction_ms = model.predict(input_df)[0]
    return round(prediction_ms, 1), condition


if __name__ == "__main__":
    rt, cond = predict_reaction_time(light_lux=1100, noise_db=68, n_back_level=2)
    print(f"Predicted reaction time: {rt} ms (condition inferred: {cond})")
