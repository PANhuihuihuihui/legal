num_train_epochs = 10
batch_size = 12
chunk_size = 256

gradient_accumulation_steps=8
fp16=True

# model_checkpoint = "zlucia/custom-legalbert"
model_checkpoint = "bert-base-multilingual-uncased"

output_dir = "/home/huijie/legal/huggface/multilingual_cail_improve"