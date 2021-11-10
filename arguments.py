import argparse

def ICTrainer_args():

    parser = argparse.ArgumentParser()
    # model params
    parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
    parser.add_argument("--intent_dropout", type=float)
    parser.add_argument("--intent_hidden", type=int)
    

    # training params
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--l2', type=float, default=0.003)
    parser.add_argument('--mode', type=str, default='BASELINE')
    parser.add_argument('--checkNEpoch', type=int, default=1)

    # data params
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--aug_dir', type=str, default=' ')
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--num_class', type=int)
    parser.add_argument('--dataset',type=str)

    #misc params
    parser.add_argument('--exp_num', type=str)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--param_save_dir', type=str)
    parser.add_argument('--logging_dir', type=str)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--freeze_args', type=bool,default=False)
    parser.add_argument('--experiment_type', type=str)

    args = parser.parse_args()

    return args


def mlm_params():

    parser = argparse.ArgumentParser()
    
    # model params
    parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
    
    # training params
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--l2', type=float, default=0.003)

    # data params
    parser.add_argument('--trainDir', type=str)
    parser.add_argument('--valDir', type=str)
    
    # misc. params
    parser.add_argument('--paramDir', type=str)
    parser.add_argument('--logDir', type=str)
    
    return parser.parse_args()


def ICTesting_args():

    parser = argparse.ArgumentParser()
    # model params
    parser.add_argument('--encoder', type=str, default='distilbert-base-cased')
    parser.add_argument('--tokenizer', type=str, default='distilbert-base-cased')
    parser.add_argument("--intent_dropout", type=float)
    parser.add_argument("--intent_hidden", type=int)
    

    # training params
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--l2', type=float, default=0.003)
    parser.add_argument('--mode', type=str, default='BASELINE')
    parser.add_argument('--checkNEpoch', type=int, default=1)

    # data params
    parser.add_argument('--test_dir', type=str)
    parser.add_argument('--num_class', type=int)
    parser.add_argument('--dataset',type=str)

    #misc params
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--freeze_args', type=bool,default=False)

    parser.add_argument('--model_dir',type=str)

    args = parser.parse_args()

    return args

