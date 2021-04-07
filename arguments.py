import argparse
   
def jointBert_argument():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--joint_loss_coef', type=float, default=0.4)
    parser.add_argument('--model_mode', type=str , default='IC_NER_MODE')
    
    #training parameters 
    parser.add_argument('--encoder_lr', type=float , default=0.0005)
    parser.add_argument('--intent_lr', type=float , default=0.002)
    parser.add_argument('--slots_lr', type=float , default=0.002)

    parser.add_argument('--epoch',type=int,default=19)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--weight_decay',type=float,default=0.003)
    parser.add_argument('--shuffle_data', type=bool , default=True)
    parser.add_argument('--num_worker', type=int , default=4)

    # data
    parser.add_argument('--train_dir',type=str)
    parser.add_argument('--val_dir',type=str)
    parser.add_argument('--intent_num', type=int, default=18)
    parser.add_argument('--slots_num', type=int , default=159)
    parser.add_argument('--max_len', type=int, default=56)

    #misc. 
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--weight_dir', type=str)
    parser.add_argument('--fix_seed', type=bool, default=False)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--accumulate_grad',type=int, default=4)

    parser.add_argument('--test_exp',type=bool, default=True)
    

    return parser.parse_args()


def test_argument():
    parser = argparse.ArgumentParser()
    
    # model arguments
    parser.add_argument('--model_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--tokenizer_name', type=str, default='distilbert-base-multilingual-cased')
    parser.add_argument('--joint_loss_coef', type=float, default=0.4)
    parser.add_argument('--model_mode', type=str , default='IC_NER_MODE')
    
    #training parameters 
    parser.add_argument('--encoder_lr', type=float , default=0.0005)
    parser.add_argument('--intent_lr', type=float , default=0.002)
    parser.add_argument('--slots_lr', type=float , default=0.002)

    parser.add_argument('--epoch',type=int,default=19)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--weight_decay',type=float,default=0.003)
    parser.add_argument('--shuffle_data', type=bool , default=True)
    parser.add_argument('--num_worker', type=int , default=4)

    # data
    parser.add_argument('--test_dir',type=str)
    parser.add_argument('--model_dir',type=str)

    #misc. 
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--test_exp',type=bool, default=True)
    

    return parser.parse_args()
