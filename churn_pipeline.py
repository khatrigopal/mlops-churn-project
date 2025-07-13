import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model_metrics import ModelMetrics, FileSource
import sagemaker
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SageMaker session and setup
region = boto3.Session().region_name
session = PipelineSession()
role = "arn:aws:iam::392928625070:role/SageMakerChurnRole"
bucket = "kg-mlops-churn-model-artifacts"

# Ensure bucket is accessible
s3 = boto3.client('s3')
try:
    s3.head_bucket(Bucket=bucket)
except Exception as e:
    logger.error(f"Bucket {bucket} not accessible: {str(e)}")
    raise

# Parameter
input_data = ParameterString(
    name="InputDataUrl",
    default_value="s3://mlops-churn-processed-data/preprocessed.csv"
)

# Preprocessing Step
sklearn_image_uri = sagemaker.image_uris.retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.xlarge"
)

script_processor = ScriptProcessor(
    image_uri=sklearn_image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    sagemaker_session=session,
    base_job_name="churn-preprocess"
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=script_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
            destination=f"s3://{bucket}/processed/train"
        ),
        ProcessingOutput(
            output_name="validation",
            source="/opt/ml/processing/output/validation",
            destination=f"s3://{bucket}/processed/validation"
        )
    ],
    code="preprocessing.py"
)

# Training Step
xgb_image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1",
    py_version="py3",
    instance_type="ml.m5.xlarge"
)

xgb_estimator = XGBoost(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    framework_version="1.5-1",
    py_version="py3",
    output_path=f"s3://{bucket}/output",
    sagemaker_session=session,
    hyperparameters={
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "seed": 42
    },
    environment={
        "MLFLOW_TRACKING_URI": "http://13.201.60.187:30081/",
        "MLFLOW_EXPERIMENT_NAME": "ChurnPrediction"
    },
    base_job_name="churn-train",
    dependencies=["requirements.txt"]
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Model Registration
model = Model(
    image_uri=xgb_image_uri,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=session,
    env={
        "MLFLOW_TRACKING_URI": "http://13.201.60.187:30081/",
        "MLFLOW_EXPERIMENT_NAME": "ChurnPrediction"
    }
)

model_metrics_report = FileSource(
    s3_uri=f"s3://{bucket}/metrics/model_metrics.json",
    content_type="application/json"
)

model_metrics = ModelMetrics()
model_metrics.model_quality = model_metrics_report

register_model_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="ChurnModelPackageGroup",
        approval_status="PendingManualApproval",
        model_metrics=model_metrics
    )
)

# Define Pipeline
pipeline = Pipeline(
    name="churn-pipeline",
    parameters=[input_data],
    steps=[processing_step, train_step, register_model_step],
    sagemaker_session=session
)

# Run Pipeline
if __name__ == "__main__":
    try:
        logger.info("Creating/updating pipeline...")
        pipeline.upsert(role_arn=role)
        logger.info("Pipeline definition upserted successfully.")
    except Exception as e:
        logger.error(f"Pipeline definition or upsert failed: {str(e)}")
        raise
