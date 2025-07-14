import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor
from transformers.models.clip import CLIPModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
import os
import argparse



class VQADataset(Dataset):
    """VQA 대조 학습을 위한 사용자 정의 데이터셋"""
    def __init__(self, df, processor, image_dir):
        self.df = df
        self.processor = processor
        self.image_dir = image_dir
        self.choice_cols = ['A', 'B', 'C', 'D']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        question = row['Question']
        correct_answer_col = row['answer']
        #row['img_path'] = ./image/train_000000.jpg
        #image_dir = /data/dacon
        # /data/dacon/image/train_000000.jpg
        img_path = os.path.join(self.image_dir, row['img_path'])
        img_path =f"/workspace/clip_vqa_finetuning/{row['img_path']}" 
        image = Image.open(img_path).convert("RGB")

        # 모든 선택지 텍스트 구성
        all_choices_texts = ["Q: " + question + " A: " + row[col] for col in self.choice_cols]
        correct_choice_idx = self.choice_cols.index(correct_answer_col)

        inputs = self.processor(
            text=all_choices_texts, 
            images=image, 
            return_tensors="pt", 
            padding='max_length', # 모델의 최대 길이에 맞춰 패딩
            truncation=True,      # 최대 길이보다 길 경우 자르기
            max_length=77         # CLIP의 표준 최대 길이
        )
        
        # 단일 이미지에 대해 프로세서가 추가한 배치 차원 제거
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['label'] = torch.tensor(correct_choice_idx, dtype=torch.long)

        return inputs
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument('--train_csv_path', type=str, default="/workspace/clip_vqa_finetuning/train_final.csv")
    parser.add_argument('--image_dir', type=str, default="/data/dacon")
    parser.add_argument('--output_dir', type=str, default="/data/dacon/result")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    return parser.parse_args()
def is_distributed():
    return dist.is_available() and dist.is_initialized()

def train(args):
    """VQA를 위한 대조 학습을 사용하여 CLIP 모델을 파인튜닝합니다."""
    # 1. 설정
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        print(f"[GPU {local_rank}] 분산 학습 준비 완료")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[싱글 GPU 또는 CPU] 장치 설정 완료: {device}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. 모델 및 프로세서 로드
    print(f"모델 로드 중: {args.model_name}")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/data/dacon/hf_cache").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/data/dacon/hf_cache")
    if is_distributed():
            model = DDP(model, device_ids=[local_rank])

    # 3. 데이터 로드
    print(f"데이터 로드 중: {args.train_csv_path}")
    if not os.path.exists(args.train_csv_path):
        print(f"오류: {args.train_csv_path} 에서 학습 데이터를 찾을 수 없습니다.")
        return
        
    train_df = pd.read_csv(args.train_csv_path)
    dataset = VQADataset(df=train_df, processor=processor, image_dir=args.image_dir)
    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # 4. 옵티마이저
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # 5. 학습 루프
    print("학습을 시작합니다...")
    model.train() # 모델을 학습 모드로 설정

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 에포크마다 셔플 재설정
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"[GPU {local_rank}] 에포크 {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")

            input_ids = batch["input_ids"].view(-1, batch["input_ids"].shape[-1])
            attention_mask = batch["attention_mask"].view(-1, batch["attention_mask"].shape[-1])
            pixel_values = batch["pixel_values"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

            logits = outputs.logits_per_image
            batch_size = pixel_values.size(0)
            num_choices = 4
            logits_for_loss = torch.zeros(batch_size, num_choices).to(device)
            for i in range(batch_size):
                logits_for_loss[i] = logits[i, i*num_choices : (i+1)*num_choices]

            loss = torch.nn.functional.cross_entropy(logits_for_loss, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if dist.get_rank() == 0:  # 마스터 프로세스만 출력
            print(f"[GPU {local_rank}] 에포크 {epoch+1} 완료. 평균 손실: {avg_loss:.4f}")

    # 6. 모델 저장
    if dist.get_rank() == 0:
        model.module.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("모델 저장 완료.")



if __name__ == "__main__":
    args = parse_args()
    train(args)
    
    