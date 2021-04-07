from jointBert import *
from os import listdir
from os.path import isfile, join
args = test_argument()

def cal_mean_stderror(metric):
    var,std_error = 0,0
    mean = sum(metric)/len(metric)
    for m in metric:
        var += (m-mean)**2
    var = (var/(len(metric)-1))**0.5
    std_error = var/((len(metric))**0.5)
    return [round(mean,4),round(std_error,4)]

# fetching slot-idx dictionary
final_slots = pd.read_csv('./data/multiATIS/slots_list.csv',sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
idx2slots  = {idx:slots for idx,slots in enumerate(final_slots)}

test_dir = args.test_dir 
testSet_Path = [f for f in listdir(test_dir) if isfile(join(test_dir, f))]

model_dir = args.model_dir 
model_Path = [f for f in listdir(model_dir) if isfile(join(model_dir, f))]

trainer = pl.Trainer(gpus=-1,precision=16,accumulate_grad_batches=4,max_epochs=15, check_val_every_n_epoch=1)

for test_file in testSet_Path:

    dl = NLU_Dataset_pl(test_file,test_file,test_file , 'distilbert-base-multilingual-cased',56,1)
    dl.setup()
    testLoader = dl.test_dataloader()

    acc,slotF1 = [],[]

    for model_path in model_Path:

        model = jointBert.load_from_checkpoint(checkpoint_path=model_Path,map_location=None)
        model.eval()

        out = trainer.test(model=model,test_dataloaders=testLoader)

        acc.append(out[0]['test_intent_acc'])
        slotF1.append(out[0]['test_slot_f1'])
    

    print('test_file: ', test_file ,'acc:',cal_mean_stderror(acc),'slotsF1',cal_mean_stderror(slotF1))
