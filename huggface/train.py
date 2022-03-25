
from tqdm.auto import tqdm
import torch
import math
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from transformers import AutoModelForMaskedLM

from dataloader import getDataloader,getDataset
import config

model = AutoModelForMaskedLM.from_pretrained(config.model_checkpoint)
# datasets

hklii_dataset,eval_dataset = getDataset()
# dataloader
train_dataloader,eval_dataloader = getDataloader(hklii_dataset,eval_dataset)
print("got dataloader")
# training
model = AutoModelForMaskedLM.from_pretrained(config.model_checkpoint)

optimizer = AdamW(model.parameters(), lr=5e-5)


accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = config.num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
## get the perplexity first
log = []

model.eval()
losses = []
for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)

    loss = outputs.loss
    losses.append(accelerator.gather(loss.repeat(config.batch_size)))

losses = torch.cat(losses)
losses = losses[: len(eval_dataset)]
try:
    perplexity = math.exp(torch.mean(losses))
except OverflowError:
    perplexity = float("inf")
log.append("model_name: {}".format(config.model_checkpoint))
log.append("Perplexity (before): {}".format(perplexity))
print(log[-1])




# training
progress_bar = tqdm(range(num_training_steps),ncols = 25 )



for epoch in range(config.num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(config.batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
    log.append(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(config.output_dir, save_function=accelerator.save)
    # if accelerator.is_main_process:
    #     tokenizer.save_pretrained(config.output_dir)
    #     repo.push_to_hub(
    #         commit_message=f"Training in progress epoch {epoch}", blocking=False
    #     )

print("\n".join(log))