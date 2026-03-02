"""
Entry point for PMD + Tool-Integrated Reasoning (TIR) training.

Identical to openkimi.pmd.main_pmd, except:
- Uses TIRPMDTrainer (supports mask_void_turns).
- Imports TIRAgentLoop to register the "tir_agent" agent loop name.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available

# Register TIR components in this process.
import examples.tir.tir_agent_loop  # noqa: F401
import examples.tir.tir_reward_manager  # noqa: F401

# Import PMD algorithm registrations (ploo advantage estimator, opmd loss).
from openkimi.pmd import core_algos  # noqa: F401
from openkimi.pmd.main_pmd import TaskRunner
from examples.tir.tir_pmd_trainer import TIRPMDTrainer


class TIRTaskRunner(TaskRunner):
    """TaskRunner that uses TIRPMDTrainer instead of RayPMDTrainer."""

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker and register PMD + TIR agent loop in worker processes."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # Use new model engine implementation.
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            class TIRActorRolloutRefWorker(ActorRolloutRefWorker):
                def __init__(self, *args, **kwargs):
                    import openkimi.pmd.core_algos  # noqa: F401
                    import examples.tir.tir_agent_loop  # noqa: F401
                    super().__init__(*args, **kwargs)

            actor_rollout_cls = TIRActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            class TIRAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
                def __init__(self, *args, **kwargs):
                    import openkimi.pmd.core_algos  # noqa: F401
                    import examples.tir.tir_agent_loop  # noqa: F401
                    super().__init__(*args, **kwargs)

            actor_rollout_cls = TIRAsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            class TIRAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
                def __init__(self, *args, **kwargs):
                    import openkimi.pmd.core_algos  # noqa: F401
                    import examples.tir.tir_agent_loop  # noqa: F401
                    super().__init__(*args, **kwargs)

            actor_rollout_cls = TIRAsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def run(self, config):
        from pprint import pprint
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        import examples.tir.tir_agent_loop  # noqa: F401
        from examples.tir.tir_reward_manager import TIRDAPORewardManager
        from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY

        REWARD_MANAGER_REGISTRY.setdefault("tir_dapo", TIRDAPORewardManager)
        try:
            from verl.experimental.reward_loop.reward_manager.registry import REWARD_LOOP_MANAGER_REGISTRY

            REWARD_LOOP_MANAGER_REGISTRY.setdefault("tir_dapo", TIRDAPORewardManager)
        except Exception:
            pass

        print(f"TIRTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = TIRDAPORewardManager(config=config, tokenizer=tokenizer)
        val_reward_fn = TIRDAPORewardManager(config=config, tokenizer=tokenizer)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = TIRPMDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)

    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**OmegaConf.to_container(ray_init_kwargs), "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(TIRTaskRunner)
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()

    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
