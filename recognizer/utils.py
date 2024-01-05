import os

def verbosity() -> int:
    if env_verbosity := os.getenv("RECOGNIZER_VERBOSE"):
        try:
            return int(env_verbosity)
        except ValueError:
            print(f"Env: RECOGNIZER_VERBOSE could not be parsed to int")
            return 0
    return 0