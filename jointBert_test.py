from jointBert import *


args = jointBert_argument()

if args.fix_seed:
    seed_everything(42)

if args.device == 'cuda':
    gpus = -1
else:
    gpus = 0

# setting up logger
tb_logger = pl_loggers.TensorBoardLogger(args.log_dir)

# fetching slot-idx dictionary
final_slots = pd.read_csv('./data/multiATIS/slots_list.csv',sep=',',header=None,names=['SLOTS']).SLOTS.values.tolist()
idx2slots  = {idx:slots for idx,slots in enumerate(final_slots)}

# setting checkpoint callbacks
checkpoint_callback = ModelCheckpoint(dirpath=args.weight_dir,monitor='val_IC_NER_loss', mode='min', filename='jointBert-{epoch:02d}-{val_loss}')

