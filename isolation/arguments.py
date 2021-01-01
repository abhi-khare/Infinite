import argparse

def ICPT_arguments():
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--lr',type=float,default=0.00001)
    parser.add_argument('--weights',type=float,default=0.0001)
    parser.add_argument('--margin',type=float)
    parser.add_argument('--model_mode', type=str , default='INTENT_CONTRA_MODE')

    # data args
    parser.add_argument('--max_len',type=int,default=56)
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--intent_num', type=int, default=17)
    parser.add_argument('--slots_num', type=int , default=160)

    # training args
    parser.add_argument('--epoch',type=int,default=5)

    # misc. args
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_worker', type=int , default=4)
    parser.add_argument('--model_export', type=str)
    parser.add_argument('--exp_name',type=str)
    
    return parser.parse_args()


def IC_NER_argument():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--slots_dropout_val', type=float, default=0.1)
    parser.add_argument('--intent_dropout_val', type=float, default=0.1)
    parser.add_argument('--joint_loss_coef', type=float, default=0.5)
    parser.add_argument('--freeze_encoder', type=bool , default=False)
    parser.add_argument('--model_mode', type=str , default='IC_NER_MODE')
    
    #training parameters 
    parser.add_argument('--encoder_lr', type=float , default=0.0005)
    parser.add_argument('--intent_lr', type=float , default=0.002)
    parser.add_argument('--slots_lr', type=float , default=0.002)
    parser.add_argument('--epoch',type=int,default=35)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--weight_decay',type=float,default=0.003)
    parser.add_argument('--shuffle_data', type=bool , default=True)
    parser.add_argument('--num_worker', type=int , default=4)

    # data
    parser.add_argument('--train_dir',type=str)
    parser.add_argument('--val_dir',type=str)
    parser.add_argument('--intent_num', type=int, default=17)
    parser.add_argument('--slots_num', type=int , default=160)
    parser.add_argument('--max_len', type=int, default=56)

    #misc. 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp_name', type=str)

    return parser.parse_args()