from pathlib import Path
from typing import Callable, Optional
import os
import uvicorn

import pytorch_lightning as pl
from typing_extensions import Annotated

from nemo.collections.llm.utils import Config, task
from nemo.lightning import AutoResume, MegatronStrategy, NeMoLogger, OptimizerModule, Trainer, io, teardown
from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.deploy import DeployPyTriton


@task(namespace="llm")
def train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    log: Annotated[Optional[NeMoLogger], Config[NeMoLogger]] = None,
    resume: Annotated[Optional[AutoResume], Config[AutoResume]] = None,
    optim: Optional[OptimizerModule] = None,
    tokenizer: Optional[str] = None,
    # TODO: Fix export export: Optional[str] = None,
) -> Path:
    """
    Trains a model using the specified data and trainer, with optional tokenizer, source, and export.

    Args:
        model (pl.LightningModule): The model to be trained.
        data (pl.LightningDataModule): The data module containing training data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        log (NeMoLogger): A nemologger instance.
        resume (Optional[Union[AutoResume, Resume]]): Resume training from a checkpoint.
        optim (Optional[OptimizerModule]): The optimizer module to be used. If not provided, the default optimizer
            from the model will be used.
        tokenizer (Optional[str]): Tokenizer setting to be applied. Can be 'data' or 'model'.
        export (Optional[str]): Filename to save the exported checkpoint after training.

    Returns
    -------
        Path: The directory path where training artifacts are saved.

    Raises
    ------
        ValueError: If the trainer's strategy is not MegatronStrategy.

    Examples
    --------
        >>> model = MyModel()
        >>> data = MyDataModule()
        >>> trainer = Trainer(strategy=MegatronStrategy())
        >>> train(model, data, trainer, tokenizer='data', source='path/to/ckpt.ckpt', export='final.ckpt')
        PosixPath('/path/to/log_dir')
    """
    _log = log or NeMoLogger()
    app_state = _log.setup(
        trainer,
        resume_if_exists=getattr(resume, "resume_if_exists", False),
        task_config=getattr(train, "__io__", None),
    )
    if resume is not None:
        resume.setup(model, trainer)
    if optim:
        optim.connect(model)
    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)

    trainer.fit(model, data)

    _log.teardown()

    return app_state.exp_dir


@task(namespace="llm")
def pretrain(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    source: Optional[str] = None,
    # export: Optional[str] = None
) -> Path:
    return train(model=model, data=data, trainer=trainer, tokenizer="data", source=source)


@task(namespace="llm")
def validate(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    tokenizer: Optional[str] = None,
    source: Optional[str] = None,
    export: Optional[str] = None,
) -> Path:
    if not isinstance(trainer.strategy, MegatronStrategy):
        raise ValueError("Only MegatronStrategy is supported")

    validate_kwargs = {}
    run_dir = Path(trainer.logger.log_dir)
    export_dir = run_dir / "export"

    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)
    if source:
        _add_ckpt_path(source, model, validate_kwargs)

    trainer.validate(model, data, **validate_kwargs)
    trainer.save_checkpoint(export_dir)
    if export:
        teardown(trainer)
        del trainer, model, data
        export_ckpt(export_dir, export)

    return run_dir

def get_trtllm_deployable(nemo_checkpoint, model_type, triton_model_repository, num_gpus, tensor_parallelism_size, pipeline_parallelism_size,
                        max_input_len, max_output_len, max_batch_size, dtype):
    if triton_model_repository is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = triton_model_repository

    if nemo_checkpoint is None and triton_model_repository is None:
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint or a TensorRT-LLM engine."
        )

    if nemo_checkpoint is None and not os.path.isdir(triton_model_repository):
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint or a valid TensorRT-LLM engine."
        )

    if nemo_checkpoint is not None and model_type is None:
        raise ValueError("Model type is required to be defined if a nemo checkpoint is provided.")

    trt_llm_exporter = TensorRTLLM(
        model_dir=trt_llm_path,
        load_model=(nemo_checkpoint is None),
    )

    if nemo_checkpoint is not None:
        try:
            #LOGGER.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=nemo_checkpoint,
                model_type=model_type,
                n_gpus=num_gpus,
                tensor_parallelism_size=tensor_parallelism_size,
                pipeline_parallelism_size=pipeline_parallelism_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                dtype=dtype,
            )
        except Exception as error:
            raise RuntimeError("An error has occurred during the model export. Error message: " + str(error))

    return trt_llm_exporter

@task(namespace="llm")
def deploy(
    nemo_checkpoint: Path = None,
    model_type: str = "llama",
    triton_model_name: str,
    triton_model_version: Optional[int] = 1,
    triton_port: int = 8000,
    triton_http_address: str = "0.0.0.0",
    triton_model_repository: Path = None,
    num_gpus: int = 1,
    tensor_parallelism_size: int = 1,
    pipeline_parallelism_size: int = 1,
    dtype: str = "bfloat16",
    max_input_len: int = 256,
    max_output_len: int = 256,
    max_batch_size: int = 8,
    start_rest_service: bool = False,
    rest_service_http_address: str = "0.0.0.0",
    rest_service_port: int = 8000
):
    triton_deployable = get_trtllm_deployable(nemo_checkpoint, model_type, triton_model_repository, num_gpus,
                            tensor_parallelism_size, pipeline_parallelism_size, max_input_len, max_output_len,
                            max_batch_size, dtype)

    if start_rest_service:
        if triton_port == rest_service_port:
            raise ValueError("REST service port and Triton server port cannot use the same port.")
            return

    try:
        nm = DeployPyTriton(
            model=triton_deployable,
            triton_model_name=triton_model_name,
            triton_model_version=triton_model_version,
            max_batch_size=max_batch_size,
            port=triton_port,
            address=triton_http_addresss
        )

        print("Triton deploy function will be called.")
        nm.deploy()
    except Exception as error:
        print("Error message has occurred during deploy function. Error message: " + str(error))
        return

    try:
        print("Model serving on Triton is will be started.")
        if start_rest_service:
            try:
                print(("REST service will be started."))
                uvicorn.run(
                    'nemo.deploy.service.rest_model_api:app',
                    host=rest_service_http_address,
                    port=rest_service_port,
                    reload=True,
                )
            except Exception as error:
                print("Error message has occurred during REST service start. Error message: " + str(error))
        nm.serve()
    except Exception as error:
        print("Error message has occurred during deploy function. Error message: " + str(error))
        return

    print("Model serving will be stopped.")
    nm.stop()

@task(name="import", namespace="llm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    return io.import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite)


def load_connector_from_trainer_ckpt(path: Path, target: str) -> io.ModelConnector:
    return io.load_context(path).model.exporter(target, path)


@task(name="export", namespace="llm")
def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], io.ModelConnector] = load_connector_from_trainer_ckpt,
) -> Path:
    return io.export_ckpt(path, target, output_path, overwrite, load_connector)


def _use_tokenizer(model: pl.LightningModule, data: pl.LightningDataModule, tokenizer: str) -> None:
    if tokenizer == "data":
        model.tokenizer = data.tokenizer
        if hasattr(model, "__io__"):
            model.__io__.tokenizer = data.tokenizer
    elif tokenizer == "model":
        data.tokenizer = model.tokenizer
        if hasattr(data, "__io__"):
            data.__io__.tokenizer = model.tokenizer


def _add_ckpt_path(source, model, kwargs) -> None:
    if io.is_distributed_ckpt(source):
        kwargs["ckpt_path"] = source
    else:
        kwargs["ckpt_path"] = model.import_ckpt(source)


def _save_config_img(*args, **kwargs):
    try:
        from nemo_sdk.utils import save_config_img

        save_config_img(*args, **kwargs)
    except ImportError:
        pass
