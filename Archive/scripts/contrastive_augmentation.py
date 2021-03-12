from transformers import MarianMTModel , MarianTokenizer
import pandas as pd

device = 'cuda'

def chunks(l, n):   
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

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

text , aug_1 , aug_2 , aug_3 = [],[],[], []

# augmentation of en data
data_EN = pd.read_csv('./data/splits/text/train_EN.tsv',sep='\t')
text_EN = list(data_EN.utterance)

text  += text_EN
aug_1 += translate(text_EN,'en','fr',device)
aug_2 += translate(text_EN,'en','de',device)
aug_3 += translate(text_EN,'en','es',device)

print('Augmentation of EN data finished')

# augmentation of de data
data_DE = pd.read_csv('./data/splits/text/train_DE.tsv',sep='\t')
text_DE = list(data_DE.utterance)

text  += text_DE
aug_1 += translate(text_DE,'de','fr',device)
aug_2 += translate(text_DE,'de','en',device)
aug_3 += translate(text_DE,'de','es',device)

print('Augmentation of DE data finished')

# augmentation of en data
data_FR = pd.read_csv('./data/splits/text/train_FR.tsv',sep='\t')
text_FR = list(data_FR.utterance)

text  += text_FR
aug_1 += translate(text_FR,'fr','en',device)
aug_2 += translate(text_FR,'fr','de',device)
aug_3 += translate(text_FR,'fr','es',device)

print('Augmentation of FR data finished')

# augmentation of en data
data_ES = pd.read_csv('./data/splits/text/train_ES.tsv',sep='\t')
text_ES = list(data_ES.utterance)

text  += text_ES
aug_1 += translate(text_ES,'es','fr',device)
aug_2 += translate(text_ES,'es','de',device)
aug_3 += translate(text_ES,'es','en',device)

print('Augmentation of ES data finished')

labels = list(range(len(text)))

augDataPD = pd.DataFrame({'text':text , 'aug_1':aug_1 , 'aug_2':aug_2 , 'aug_3':aug_3 , 'label':labels}) 

augDataPD.to_csv('./data/splits/text/contrastive_data.tsv',sep='\t',index=False)