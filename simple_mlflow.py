import mlflow
import mlflow.pytorch


# How to use MLflow in a basic way :
mlflow.start_run() #Start a run
mlflow.log_param("my", "param") # Log the parameters of the run
mlflow.log_metric("score", 100) # Log the metric of the run
mlflow.end_run() # End the run

########################## OR  ############################

# Log PyTorch model
with mlflow.start_run() as run:
    mlflow.log_param("my", "param") # Log the parameters of the run
    mlflow.log_metric("score", 100) # Log the metric of the run
    mlflow.pytorch.log_model(model, "model") # Using pytorch, log a model in mlflow