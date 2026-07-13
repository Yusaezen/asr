# Lazy imports — don't import scorer at module load time to avoid
# circular import chain when train.py imports uhead.model directly.
