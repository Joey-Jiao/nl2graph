from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


class TimeoutError(Exception):
    pass


def with_timeout(config_path: str):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            timeout = None
            try:
                from .context import get_context
                from .configs import ConfigService
                config = get_context().resolve(ConfigService)
                timeout = config.get(config_path)
            except:
                pass
            if timeout is None:
                return method(self, *args, **kwargs)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(method, self, *args, **kwargs)
                try:
                    return future.result(timeout=timeout)
                except FuturesTimeoutError:
                    raise TimeoutError(f"timeout after {timeout}s")
        return wrapper
    return decorator
