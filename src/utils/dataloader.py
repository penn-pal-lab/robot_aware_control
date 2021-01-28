def inf_loop_dataloader(loader):
    loader_it = iter(loader)
    while True:
        yield next(loader_it)