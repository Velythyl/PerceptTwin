import multiprocessing
processes = multiprocessing.cpu_count()

objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=processes
)