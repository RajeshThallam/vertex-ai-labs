import sys
import os
import time
import argparse
import subprocess
import logging
from pathlib import Path
import json
import asyncio
import socket

import xgboost as xgb
from xgboost.dask import DaskDMatrix, DaskDeviceQuantileDMatrix

import dask
import dask.dataframe as dd
import dask_cudf as cudf
from dask.distributed import Client, wait

dask.config.set({"distributed.comm.timeouts.connect": "60s"})

def get_args():
    """Define the task arguments with the default values.
    Returns:
        experiment parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-dir', 
        default=os.getenv('AIP_MODEL_DIR'), 
        type=str,
        help='Cloud Storage URI of a directory for saving model artifacts')
    parser.add_argument(
        '--train-files',
        type=str,
        help='Training files local or GCS',
        required=True)
    parser.add_argument(
        '--scheduler-ip-file',
        type=str,
        help='Scratch temp file to store scheduler ip in GCS',
        required=False)
    parser.add_argument(
        '--num-workers',
        type=int,
        help='num of workers for rabit')
    parser.add_argument(
        '--rmm-pool-size',
        type=str,
        help='RMM pool size',
        default='8G')
    parser.add_argument(
        '--nthreads',
        type=str,
        help='nthreads for master and worker',
        default='4')
    parser.add_argument(
        '--parquet',
        action='store_false',
        help='parquet files are used')
    
    return parser.parse_args()

async def start_client(
    scheduler_addr, 
    train_dir, 
    num_workers, 
    gpu_mode=True,
    do_wait=False, 
    parquet=False):
    """
    """
    async with Client(scheduler_addr, asynchronous=True) as client:
        # wait until all workers are up and running
        dask.config.set({'distributed.scheduler.work-stealing': False})
        dask.config.set({'distributed.scheduler.bandwidth': 1})
        logging.info(f'distributed.scheduler.work-stealing={dask.config.get("distributed.scheduler.work-stealing")}')
        logging.info(f'distributed.scheduler.bandwidth={dask.config.get("distributed.scheduler.bandwidth")}')
        await client.wait_for_workers(num_workers)

        # read dataframe
        colnames = ['label'] + ['feature-%02d' % i for i in range(1, 29)]

        # read as csv or parquet
        if parquet is True:
            if gpu_mode:
                df = cudf.read_parquet(train_dir, columns=colnames)
            else:
                df = dd.read_parquet(train_dir, columns=colnames)
        else:
            if gpu_mode:
                df = cudf.read_csv(train_dir, header=None, names=colnames, chunksize=None)
            else:
                df = dd.read_csv(train_dir, header=None, names=colnames, chunksize=None)

        # get features and target label
        X = df[df.columns.difference(['label'])]
        y = df['label']
        
        # wait for fully computing results
        if do_wait is True:
            df = df.persist()
            X = X.persist()
            wait(df)
            wait(X)
            logging.info("[debug:leader]: ------ Long waited but the data is ready now")

        # compute DMatrix for training xgboost
        # for GPU compute DaskDeviceQuantileDMatrix
        start_time = time.time()
        if gpu_mode:
            dtrain = await DaskDeviceQuantileDMatrix(client, X, y)
        else:
            dtrain = DaskDMatrix(client, X, y)
        logging.info("[debug:leader]: ------ QuantileDMatrix is formed in {} seconds ---".format((time.time() - start_time)))

        # remove data from distributed RAM by removing the collection from local process
        del df
        del X
        del y

        # start training
        logging.info("[debug:leader]: ------ training started")
        start_time = time.time()
        xgb_params = {
            'verbosity': 2,
            'learning_rate': 0.1,
            'max_depth': 8,
            'objective': 'reg:squarederror',
            'subsample': 0.6,
            'gamma': 1,
            'verbose_eval': True,
            'tree_method': 'gpu_hist' if gpu_mode else 'hist',
            'nthread': 1
        }
        output = await xgb.dask.train(
            client,
            xgb_params,
            dtrain,
            num_boost_round=100, 
            evals=[(dtrain, 'train')])
        logging.info("[debug:leader]: ------ training finished")

        # evaluation history
        history = output['history']
        logging.info('[debug:leader]: ------ Training evaluation history:', history)

        # save model
        model_file = f"{model_dir}/model/xgboost.model"
        output['booster'].save_model(model_file)
        logging.info(f"[debug:leader]: ------model saved {model_file}")

        logging.info("[debug:leader]: ------ %s seconds ---" % (time.time() - start_time))

        # wait for client to shutdown
        await client.shutdown()

def launch_dask(cmd, is_shell):
    """ launch dask scheduler
    """
    return subprocess.Popen(cmd, stdout=None, stderr=None, shell=is_shell)

def launch_worker(cmd):
    """ launch dask workers
    """
    return subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)

def get_scheduler_ip(scheduler_ip_file):
    with open(scheduler_ip_file, 'r') as f:
        scheduler_ip = f.read().rstrip("\n")
    return scheduler_ip

if __name__=='__main__':
    logging.basicConfig(format='%(message)s')
    logging.getLogger().setLevel(logging.INFO)

    # get program args
    args = get_args()

    # set and create local directories if does not exists
    local_tmp_dir = os.path.join(os.getcwd(), "tmp")
    Path(local_tmp_dir).mkdir(parents=True, exist_ok=True)
    local_model_dir = os.path.join(local_tmp_dir, 'model')
    Path(local_model_dir).mkdir(parents=True, exist_ok=True)

    # define variables
    gs_prefix = 'gs://'
    gcsfuse_prefix = '/gcs/'

    logging.info(f'[INFO]: args.model_dir = {args.model_dir}')

    model_dir = args.model_dir or local_model_dir
    if model_dir and model_dir.startswith(gs_prefix):
        model_dir = model_dir.replace(gs_prefix, gcsfuse_prefix)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    tmp_dir = model_dir or local_tmp_dir
    if not tmp_dir.startswith(gs_prefix):
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    scheduler_ip_file = f"{tmp_dir}dask_scheduler.txt" if tmp_dir[-1] == "/" else f"{tmp_dir}/dask_scheduler.txt"

    logging.info(f'[INFO]: model_dir = {model_dir}')
    logging.info(f'[INFO]: tmp_dir = {tmp_dir}')
    logging.info(f'[INFO]: scheduler_ip_file = {scheduler_ip_file}')

    # read worker pool config and launch dask scheduler and workers
    TF_CONFIG = os.environ.get('TF_CONFIG')
    
    if TF_CONFIG:
        TF_CONFIG = json.loads(TF_CONFIG)
        logging.info(TF_CONFIG)
        task_name = TF_CONFIG.get('task', {}).get('type')
    else:
        logging.info(f'Running locally')
        task_name = 'chief'

    scheduler_port = '8786'
        
    if task_name == 'chief':
        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)

        with open(scheduler_ip_file, 'w') as f:
            f.write(host_ip)

        scheduler_addr = f'{host_ip}:{scheduler_port}'
        logging.info('[INFO]: The scheduler IP is %s', scheduler_addr)
        proc_scheduler = launch_dask(f'dask-scheduler --protocol tcp > {tmp_dir}/scheduler.log 2>&1 &', True)
        logging.info('[debug:leader]: ------ start scheduler')

        proc_worker = launch_dask([
            'dask-cuda-worker', 
            '--rmm-pool-size', args.rmm_pool_size, 
            '--nthreads', args.nthreads,
            scheduler_addr], 
            False)
        logging.info('[debug:leader]: ------ start worker')
        asyncio.get_event_loop().run_until_complete(
            start_client(
                scheduler_addr,
                args.train_files,
                args.num_workers,
                parquet=False))

    # launch dask worker, redirect output to sys stdout/err
    elif task_name == 'worker':
        while not os.path.isfile(scheduler_ip_file):
            time.sleep(1)
        
        # with open(scheduler_ip_file, 'r') as f:
        #     scheduler_ip = f.read().rstrip("\n")
        scheduler_ip = get_scheduler_ip(scheduler_ip_file)
        while not scheduler_ip:
            time.sleep(1)
            scheduler_ip = get_scheduler_ip(scheduler_ip_file)

        scheduler_ip = get_scheduler_ip(scheduler_ip_file)
        logging.info(f'[debug:scheduler_ip]: ------ {scheduler_ip}')
        scheduler_addr = f'{scheduler_ip}:{scheduler_port}'

        proc_worker = launch_worker([
            'dask-cuda-worker', 
            '--rmm-pool-size', args.rmm_pool_size, 
            '--nthreads' , args.nthreads, 
            scheduler_addr])