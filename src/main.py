from src.preprocessing import load_and_merge, clean_data, feature_engineering, split_data
from src.model import train_model, model_prediction
from src.utils import load_params
from src.model_serializer import save_model
from src.evaluation import get_evaluations
from src.data_loader import load_and_merge
from src.optimization import tune_hyperparameters

config = load_params()
train_dataset = config['paths']['train_dataset']
store_dataset = config['paths']['store_dataset']
THRESHOLD = config['PERFORMANCE_THRESHOLD_RMSPE']

def run_pipeline():
    
    print("Step 1: Loading Data")
    df = load_and_merge(train_dataset, store_dataset)

    print("Step 2: Cleaning & Engineering")
    df = clean_data(df)
    df = feature_engineering(df)

    print("Step 3: Splitting")
    X_train, y_train, X_cv, y_cv, X_test, y_test = split_data(df)

    print("Step 4: Training Model")
    model_type = config['models']['type']
    print(f"\nModel Type: {model_type}\n")
    model = train_model(X_train, y_train, model_type)

    print("Step 5: Model Predicting before tuning")
    y_pred_default = model_prediction(model, X_cv)

    print("Step 6: Evaluating before Tuning")
    errors_before = get_evaluations(y_true=y_cv, y_pred=y_pred_default)
    
    best_model = model
    tuned_model = None

    if errors_before['RMSPE'] <= THRESHOLD:
        print(f"Skipping Step 7 i.e. hyperparameter tuning.Error ({errors_before['RMSPE']}) is already below the threshold ({THRESHOLD}).")
        best_model = model
        errors_after = errors_before
    else:
        print(f"Error {errors_before['RMSPE']} is above the threshold {THRESHOLD}. Hyperparameters need to be tuned.")
        print("Step 7: Tuning the hyperparameters")
        tuned_model = tune_hyperparameters(X_train, y_train, model_type, model)

        print("Step 8: Model predicting after Tuning")
        y_pred_tuned = model_prediction(tuned_model, X_cv)

        print("Step 9: Evaluating after Tuning")
        errors_after = get_evaluations(y_true=y_cv, y_pred=y_pred_tuned)

        if errors_after['RMSPE'] <= errors_before['RMSPE']:
            print(f"Tuning reduced the error ({errors_after['RMSPE'] <= errors_before['RMSPE']}). Keeping tuned version of the model.")
            best_model = tuned_model
        else:
            print(f"Tuning went wrong. The error further increased ({errors_after['RMSPE'] > errors_before['RMSPE']}). Kepping the baseline model.")
            best_model = model

    if best_model == tuned_model:
        if errors_after['RMSPE'] <= THRESHOLD:
            print(f"After tuning, error({errors_after['RMSPE']}) is below the threshold ({THRESHOLD}) now. Model is ready to be saved.")
        else:
            print(f"Error ({errors_after['RMSPE']}) reduced but it is not below the threshold yet. Maybe another iteration of tuning can help reduce it further.")
            print("WARNING: Final model has error still above the threshold.")

    elif best_model == model:
        if errors_before['RMSPE'] > THRESHOLD:
            print(f"WARNING: Final model has error ({errors_before['RMSPE']}) still above the threshold ({THRESHOLD}).")
            print("Recommendation: Get more data or engineer new features.")

    print("Step 10: Final Evaluation on Test Set")
    y_pred_final = model_prediction(best_model, X_test)
    get_evaluations(y_true=y_test, y_pred=y_pred_final)

    print("Step 11: Saving the Model")
    save_model(best_model, model_type)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()