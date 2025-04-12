import os
import time
import json
import contextlib
import datetime
import random
import torch
from .logger import log, console
from .util import TrainArgs, set_path, clean_dict, free, login, info
from .config import get_config
from .caption import tags, prompt


login()
def set_config(args: TrainArgs):
    from modules.util.config.TrainConfig import TrainConfig # pylint: disable=import-error
    config = get_config('onetrainer')

    if args.type:
        if args.type == 'sdxl':
            config['model_type'] = "STABLE_DIFFUSION_XL_10_BASE"
            config['unet']['train'] = True
            config['prior']['train'] = False
        elif args.type == 'sd':
            config['model_type'] = 'STABLE_DIFFUSION_15'
            config['unet']['train'] = True
            config['prior']['train'] = False
        elif args.type == 'flux':
            config['model_type'] = 'FLUX_DEV_1'
            config['unet']['train'] = False
            config['prior']['train'] = True
            config['prior']['weight_dtype'] = "FLOAT_8"
            config['text_encoder_2']['weight_dtype'] = "FLOAT_8"
        else:
            log.warning(f'Unknown Model type: {args.type}')
    if args.model:
        config['base_model_name'] = args.model
    if args.optimizer:
        config['optimizer']['optimizer'] = args.optimizer
    if args.d:
        config['optimizer']['d_coef'] = args.d
    if args.scheduler:
        config['learning_rate_scheduler'] = args.scheduler
    if args.lr:
        config['learning_rate'] = args.lr
    if args.warmup:
        config['learning_rate_warmup_steps'] = args.warmup
    if args.rank:
        config['lora_rank'] = args.rank
    if args.alpha:
        config['lora_alpha'] = args.alpha
    if args.batch:
        config['batch_size'] = args.batch
    if args.accumulation:
        config['gradient_accumulation_steps'] = args.accumulation
    if args.resolution:
        config['resolution'] = str(args.resolution)
    if args.epochs:
        config['epochs'] = args.epochs
    if args.triton:
        config['optimizer.use_triton'] = True
    if args.resume:
        config['continue_last_backup'] = True
    if args.te:
        config['text_encoder']['train'] = True
    if args.nobias:
        config['optimizer']['use_bias_correction'] = False
    if args.nogradient:
        config['gradient_checkpointing'] = False
    if args.interval is None:
        args.interval = 0
    if args.backup and args.interval > 0:
        config['backup_after'] = args.interval
        config['backup_after_unit'] = 'EPOCH'
    if args.save and args.interval > 0:
        config['save_after'] = args.interval
        config['save_after_unit'] = 'EPOCH'
        config['save_filename_prefix'] = args.concept
    if args.sample and args.interval > 0:
        config['sample_after'] = args.interval
        config['sample_after_unit'] = 'EPOCH'
    os.makedirs(args.tmp, exist_ok=True)
    config['debug_dir'] = os.path.join(args.tmp, 'debug')
    config['workspace_dir'] = os.path.join(args.tmp, 'workspace')
    config['cache_dir'] = os.path.join(args.tmp, 'cache')
    config['concept_file_name'] = os.path.join(args.tmp, 'concept.json')
    config['output_model_destination'] = args.output or os.path.join(args.tmp, f'{args.concept}.safetensors')
    config['sample_definition_file_name'] = os.path.join(args.tmp, 'samples.json')

    train_config = TrainConfig.default_values()
    train_config.from_dict(config)

    with open(os.path.join(args.tmp, 'config.json'), "w", encoding='utf-8') as f:
        log.info(f'write train config: file="{os.path.join(args.tmp, "config.json")}"')
        json.dump(config, f, indent=2)

    with open(train_config.concept_file_name, "w", encoding='utf-8') as f:
        concepts = get_config('concepts')
        concepts[0]["name"] = args.concept
        concepts[0]["path"] = args.input
        concepts[0]["text"]["prompt_path"] = args.input
        if args.resolution:
            concepts[0]["image"]["enable_resolution_override"] = True
            concepts[0]["image"]["resolution_override"] = str(train_config.resolution)
        log.info(f'write concepts config: file="{train_config.concept_file_name}" name="{args.concept}"')
        json.dump(concepts, f, indent=2)

    with open(train_config.sample_definition_file_name, "w", encoding='utf-8') as f:
        sample = get_config('sample')
        seed = sample.get('seed', int(random.randrange(4294967294)))
        width = sample.get('width', 1024)
        height = sample.get('height', 1024)
        steps = sample.get('steps', 20)
        cfg = sample.get('cfg', 7.0)
        scheduler = sample.get('scheduler', 'UNIPC')
        samples = get_config('samples')
        log.info(f'samples: enabled={args.sample} seed={seed} width={width} height={height} steps={steps} cfg={cfg} scheduler={scheduler}')
        log.info(f'samples prompt: "{prompt()}"')
        for i, _s in enumerate(samples):
            samples[i]["enabled"] = args.sample
            samples[i]["prompt"] = prompt()
            samples[i]["seed"] = seed
            samples[i]["width"] = width
            samples[i]["height"] = height
            samples[i]["diffusion_steps"] = steps
            samples[i]["cfg_scale"] = cfg
            samples[i]["noise_scheduler"] = scheduler
        log.info(f'write samples config: file="{train_config.sample_definition_file_name}"')
        json.dump(samples, f, indent=2)

    token = os.environ.get('HF_TOKEN', None)
    if token is not None:
        hf.login(token=token, add_to_git_credential=False, write_permission=False)

    return train_config, config


def train(args: TrainArgs):
    if not args.train:
        return
    free()

    info.status = 'start'
    set_path(args)

    from .logger import pbar
    from modules.util.callbacks.TrainCallbacks import TrainCallbacks # pylint: disable=import-error
    from modules.util.commands.TrainCommands import TrainCommands # pylint: disable=import-error
    from modules.trainer.GenericTrainer import GenericTrainer # pylint: disable=import-error
    from modules.module.BaseImageCaptionModel import BaseImageCaptionModel # pylint: disable=import-error

    def train_progress_callback(p, max_sample, max_epoch):
        ts = time.time()
        info.progress = p
        info.total = max_sample * max_epoch
        info.epoch = p.epoch
        info.step = p.global_step
        info.complete = int(100 * info.step / info.total)
        info.update = time.time()
        info.status = 'train'
        info.its = p.global_step / (ts - info.start)
        mem = torch.cuda.mem_get_info()
        info.mem = f'{1-mem[0]/mem[1]:.2f}'
        if p.global_step == 1:
            log.info(f'settings: steps={info.total} epochs={max_epoch} images={len(info.samples)} samples={max_sample}')
        if not args.nopbar:
            pbar.update(task, completed=info.step, total=info.total, description="train", text=f'step:{info.step}/{info.total} batch:{p.epoch_step} epoch:{info.epoch+1}/{max_epoch} samples:{max_sample} its:{info.its:.2f} memory:{info.mem}')

    def log_update(s: str):
        if 'loading' in s:
            info.status = 'loading'
        if s in 'sampling':
            info.status = 'sample'
            if not args.nopbar:
                pbar.update(task, completed=info.step, total=info.total, description="train", text=f'step: {info.step} epoch: {info.epoch+1} generating sample')
        elif s not in ['training', 'starting epoch/caching']:
            log.info(f'update: {s}')

    info.busy = True
    info.id = args.id
    info.concept = args.concept
    info.samples = BaseImageCaptionModel._BaseImageCaptionModel__get_sample_filenames(args.input) # pylint: disable=protected-access
    callbacks = TrainCallbacks()
    callbacks.set_on_update_status(log_update)
    callbacks.set_on_update_train_progress(train_progress_callback)
    commands = TrainCommands()
    config, config_json = set_config(args)
    log.info(f'method={config.training_method} type={config.model_type}')
    log.info(f'model="{config.base_model_name}"')
    log.info(f'concept="{args.concept}" sample={args.sample} backup={args.backup} save={args.save} interval={args.interval} resume={args.resume}')

    trainer = GenericTrainer(config, callbacks, commands)
    if not args.nopbar:
        task = pbar.add_task(description="train", text="", total=100)
    info.start = time.time()
    info.status = 'starting'

    try:
        log.info('train: init')
        trainer.start()
        del trainer.model.model_spec.thumbnail
        trainer.model.model_spec.author = args.author
        trainer.model.model_spec.title = args.concept
        trainer.model.model_spec.date = datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
        config_metadata = json.dumps(clean_dict(config_json))
        trainer.model.model_spec.base_model = os.path.basename(config_json.get('base_model_name', 'unknown'))
        trainer.model.model_spec.config = config_metadata.replace('\\"', "").replace('"', "")
        trainer.model.model_spec.concepts = json.dumps({
            "name": args.concept,
            "images": len(info.samples),
            "buckets": info.buckets,
        })
        del trainer.model.model_spec.hash_sha256
        del trainer.model.model_spec.usage_hint
        trainer.model.model_spec.module = "networks.lora"
        trainer.model.model_spec.tags = json.dumps(tags(args))
        info.metadata = trainer.model.model_spec
        log.debug(f'metadata: {json.dumps(trainer.model.model_spec.__dict__, indent=2)}')
        log.info(f'settings: optimizer={config.optimizer.optimizer} scheduler={config.learning_rate_scheduler} rank={config.lora_rank} alpha={config.lora_alpha} batch={config.batch_size} accumulation={config.gradient_accumulation_steps} dropout={config.dropout_probability} lr={config.learning_rate} d={config.optimizer.d_coef} bias={config.optimizer.use_bias_correction}') # noqa: E501
        free()
        if hasattr(trainer.model, 'unet_lora'):
            log.info(f'lora: peft={trainer.model.unet_lora.peft_type} class={trainer.model.unet_lora.klass} train={trainer.model.train_dtype}')
        log.info('train: start')
        with pbar if not args.nopbar else contextlib.nullcontext():
            time.sleep(0.1)
            trainer.train()
        trainer.end()
    except Exception as e:
        log.error(f'train: error={e}')
        console.print_exception()

    if not args.nopbar:
        pbar.remove_task(task)
    del info.metadata
    info.busy = False
    # log.debug(f'info: {info}')
    if info.step is None or info.epoch is None:
        info.status = 'failed'
        log.error('train: failed')
        return
    log.info(f'save: {trainer.config.output_model_destination}')
    if info.epoch == args.epochs:
        info.status = 'completed'
        log.info('train: completed')
    else:
        info.status = 'partial'
        log.info('train: completed partial')
