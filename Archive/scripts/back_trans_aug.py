from transformers import MarianMTModel , MarianTokenizer
import pandas as pd
from utils import chunks


device = 'cuda'

def translate(source_text,src_lang,tgt_lang,device):
    mname = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(mname)
    model.to(device)
    tok = MarianTokenizer.from_pretrained(mname)
    
    translation = []
    for src_text_list in chunks(source_text, 128):
        batch = tok.prepare_translation_batch(src_text_list).to(device)
        gen = model.generate(**batch)
        trans_batch = tok.batch_decode(gen, skip_special_tokens=True)
        translation += trans_batch
    
    return translation


data = pd.read_csv('./data/splits/text/train_lang.tsv',sep='\t')

aug_data = []

# translate english to french
EN_data = list(data[data.Language == 'en']['utterance'])
FR_data = list(data[data.Language == 'fr']['utterance'])

# direct augmentation
EN_translated_ENFR = translate(EN_data,'en','fr',device)
FR_translated_FREN = translate(EN_data,'fr','en',device)

# self augmentation
Aug_ENDE = translate(EN_data,'en','de',device)
EN_translated_DEEN = translate(Aug_ENDE,'de','en',device)

Aug_FRDE = translate(FR_data,'fr','de',device)
FR_translated_DEFR = translate(Aug_FRDE,'de','fr',device)

# one hop cross translation
EN_DE_translation = translate(EN_data,'en','de',device)
DE_FR_translation = translate(EN_DE_translation,'de','fr',device)

FR_DE_translation = translate(FR_data,'fr','de',device)
DE_EN_translation = translate(FR_DE_translation,'de','en',device)

aug_data += EN_data
aug_data += FR_data
aug_data += EN_translated_ENFR
aug_data += FR_translated_FREN
aug_data += EN_translated_DEEN
aug_data += FR_translated_DEFR
aug_data += DE_FR_translation
aug_data += DE_EN_translation

augDataPD = pd.DataFrame({'utterance':aug_data})

augDataPD.to_csv('./data/splits/text/augment_train.txt',index=False)