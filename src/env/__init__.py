# from src.env.fetch.fetch_env import FetchEnv
# from src.env.fetch.fetch_push import FetchPushEnv
def get_env(name):
    if name == "FetchPush":
        from src.env.fetch.fetch_push import FetchPushEnv
        return FetchPushEnv