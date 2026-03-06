from omegaconf import OmegaConf

# Register a custom 'if' resolver
try:
    OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
except ValueError as e:
    pass
# Usage example
#cfg = OmegaConf.create({
#    "use_gpu": True,
#    "device": "${if:${use_gpu},cuda,cpu}"
#})

try:
    OmegaConf.register_new_resolver("eq", lambda a, b: a == b)
except ValueError as e:
    pass
#cfg = OmegaConf.create({
#    "backend": "torch",
#    "device": "${if:${eq:${backend},torch},cuda,cpu}"
#})
#print(cfg.device)  # Output: cuda


#print(cfg.device)  # Output: cuda
