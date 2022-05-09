from datasets import load_dataset, Value, ClassLabel, Features
from transformers import AutoTokenizer, DataCollatorWithPadding

def get_crime_datasets(args):
    tokenizer = AutoTokenizer.from_pretrained(args.back_bone)
    def tokenize_function(examples):
        result = tokenizer([examples["begin"],examples["text"]],truncation = True,max_length = 512)
        result["input_ids"] = result["input_ids"][0]+ result["input_ids"][1]
        result["attention_mask"] = result["attention_mask"][0]+ result["attention_mask"][1]
        return result

    features = Features({"label":ClassLabel(num_classes = 29,names=['盜竊罪', '爆竊罪', '猥褻侵犯罪', '侵害人身罪', '不小心駕駛罪', '販運危險藥物罪', '入屋犯法罪', 
                '搶劫罪', '以欺騙手段取得財產罪', '販毒罪', '販運毒品罪', '普通襲擊罪', '洗黑錢罪', '串謀詐騙罪', 
                '處理贓物罪', '刑事恐嚇罪', '串謀勒索罪', '非法入境罪', '強姦罪', '非禮罪', '縱火罪', '扒竊罪', 
                '管有危險藥物罪', '危險駕駛引致他人死亡罪', '襲擊致造成身體傷害罪', '管有虛假文書罪', '使用虛假文書罪',
                '危險駕駛引致他人身體受嚴重傷害罪', '危險駕駛罪']) ,"begin": Value("string"),"text": Value("string")})
    

    labels = ['盜竊罪', '爆竊罪', '猥褻侵犯罪', '侵害人身罪', '不小心駕駛罪', '販運危險藥物罪', '入屋犯法罪', 
            '搶劫罪', '以欺騙手段取得財產罪', '販毒罪', '販運毒品罪', '普通襲擊罪', '洗黑錢罪', '串謀詐騙罪', 
            '處理贓物罪', '刑事恐嚇罪', '串謀勒索罪', '非法入境罪', '強姦罪', '非禮罪', '縱火罪', '扒竊罪', 
            '管有危險藥物罪', '危險駕駛引致他人死亡罪', '襲擊致造成身體傷害罪', '管有虛假文書罪', '使用虛假文書罪',
             '危險駕駛引致他人身體受嚴重傷害罪', '危險駕駛罪']
    label2id =  {k:v for v,k in enumerate(labels)}
    dataset = load_dataset("json", data_files="/home/huijie/legal/crime_prediction/data_prepare/crime_prediction.json",features=features)
    
    dataset = dataset.align_labels_with_mapping(label2id, "label")
    
    dataset = dataset.shuffle(seed=args.seed)["train"].train_test_split(0.1)
    tokenized_datasets = dataset.map(
        tokenize_function, batched=False, remove_columns=["begin","text"],num_proc =16
    )
    data_collator = DataCollatorWithPadding(tokenizer,padding= "max_length",max_length = 512,pad_to_multiple_of= 8)
    return tokenized_datasets["train"],tokenized_datasets["test"],data_collator