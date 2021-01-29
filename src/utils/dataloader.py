def inf_loop_dataloader(loader):
    while True:
        for x in loader:
            yield x