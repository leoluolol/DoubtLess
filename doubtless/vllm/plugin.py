"""vLLM plugin entry point (pyproject.toml, group `vllm.general_plugins`): registers the vLLM-backend architecture so
that the front end, the engine-core process and every tensor-parallel worker can build it from
hf_overrides={"architectures": ["DoubtLessContrastForCausalLM"]}."""
ARCH = "DoubtLessContrastForCausalLM"


def register():
    from vllm import ModelRegistry
    from doubtless.vllm import sched
    sched.install()
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "doubtless.vllm.model:ContrastForCausalLM")
