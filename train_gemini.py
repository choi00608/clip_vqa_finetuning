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
        # row['img_path'] = ./image/train_000000.jpg
        # image_dir = /data/dacon
        # /data/dacon/image/train_000000.jpg
        img_path = os.path.join(self.image_dir, row['img_path'])
        #img_path =f"/workspace/clip_vqa_finetuning/{row['img_path']}" 
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
    parser = argparse.ArgumentParser(description="Finetune CLIP for VQA with Distributed Training")
    parser.add_argument('--model_name', type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument('--train_csv_path', type=str, default="/workspace/clip_vqa_finetuning/train_post.csv")
    parser.add_argument('--image_dir', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="/data/dacon/result")
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6)
    # DDP를 위한 local_rank 인자 추가 (torch.distributed.launch 또는 torchrun이 자동으로 전달)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def is_distributed():
    """분산 학습 환경인지 확인합니다."""
    return dist.is_available() and dist.is_initialized()

def setup_distributed(local_rank):
    """분산 학습 환경을 설정합니다."""
    if local_rank != -1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        print(f"[GPU {local_rank}] 분산 학습 준비 완료. Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")
    else:
        print("[싱글 GPU 또는 CPU] 분산 학습이 설정되지 않았습니다.")

def cleanup():
    """분산 학습 프로세스 그룹을 정리합니다."""
    if is_distributed():
        dist.destroy_process_group()

def train(args):
    """VQA를 위한 대조 학습을 사용하여 CLIP 모델을 파인튜닝합니다."""
    # 1. 분산 학습 설정
    local_rank = args.local_rank
    setup_distributed(local_rank)
    device = torch.device("cuda", local_rank) if local_rank != -1 else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 메인 프로세스인지 확인 (로깅, 저장 등에 사용)
    is_main_process = not is_distributed() or dist.get_rank() == 0

    if is_main_process and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 2. 모델 및 프로세서 로드
    if is_main_process:
        print(f"모델 로드 중: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    if is_distributed():
        # DDP로 모델 래핑
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 3. 데이터 로드
    if is_main_process:
        print(f"데이터 로드 중: {args.train_csv_path}")
    if not os.path.exists(args.train_csv_path):
        print(f"오류: {args.train_csv_path} 에서 학습 데이터를 찾을 수 없습니다.")
        return
        
    train_df = pd.read_csv(args.train_csv_path)
    dataset = VQADataset(df=train_df, processor=processor, image_dir=args.image_dir)
    
    # [수정] 분산 학습을 위한 Sampler 설정
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None

    # [수정] DataLoader 설정: 분산 학습 시 sampler를 사용하고, shuffle은 비활성화 (sampler가 처리)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None) # sampler가 없을 때만 shuffle 활성화
    )

    # 4. 옵티마이저
    # DDP 사용 시 model.module.parameters()를 전달해야 함
    optimizer_params = model.module.parameters() if is_distributed() else model.parameters()
    optimizer = AdamW(optimizer_params, lr=args.lr)
    
    # 5. 학습 루프
    if is_main_process:
        print("학습을 시작합니다...")
    model.train()

    for epoch in range(args.epochs):
        if is_main_process:
            print(f"--- 에포크 {epoch+1}/{args.epochs} ---")
        
        # [추가] 분산 학습 시, 에포크마다 샘플러에 현재 에포크를 설정하여 데이터 셔플링이 다르게 되도록 함
        if is_distributed():
            sampler.set_epoch(epoch)

        total_loss = 0
        # 메인 프로세스에서만 tqdm 진행률 표시
        progress_bar = tqdm(dataloader, desc=f"에포크 {epoch+1} 배치", disable=not is_main_process)
        
        for batch in progress_bar:
            # 배치를 장치로 이동
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('label')

            # 입력 데이터 형태 변경
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            
            batch_size = pixel_values.shape[0]
            num_choices = input_ids.shape[1]
            
            input_ids = input_ids.view(batch_size * num_choices, -1)
            attention_mask = attention_mask.view(batch_size * num_choices, -1)
            
            # 순전파
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            logits = outputs.logits_per_image
            
            logits_for_loss = torch.zeros(batch_size, num_choices).to(device)
            for i in range(batch_size):
                logits_for_loss[i] = logits[i, i*num_choices : (i+1)*num_choices]

            # 손실 함수 계산
            loss = torch.nn.functional.cross_entropy(logits_for_loss, labels)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward() # DDP가 여기서 그래디언트를 동기화
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if is_main_process:
            print(f"에포크 {epoch+1} 완료. 평균 손실: {avg_loss:.4f}")

    # 6. 모델 저장
    if is_main_process:
        print(f"학습 완료. 모델을 {args.output_dir}에 저장합니다.")
        # [수정] DDP로 래핑된 경우, model.module을 통해 원본 모델에 접근하여 저장
        model_to_save = model.module if is_distributed() else model
        model_to_save.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("모델이 성공적으로 저장되었습니다.")

    # [추가] 분산 학습 프로세스 정리
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    train(args)