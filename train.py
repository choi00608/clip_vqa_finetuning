import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# --- 설정 ---
MODEL_NAME = "openai/clip-vit-large-patch14"
TRAIN_CSV_PATH = "train_final.csv"
IMAGE_DIR = "train/"
OUTPUT_MODEL_DIR = "./clip_finetuned"

FINETUNED_MODEL_PATH = "./clip_finetuned"
TEST_CSV_PATH = "train_final.csv"
IMAGE_DIR = "train/"
SUBMISSION_CSV_PATH = "baseline_submit_tuned.csv"

EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 5e-6

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
        
        # 이미지
        img_path = row['img_path']
        image = Image.open(img_path).convert("RGB")

        # 모든 선택지 텍스트 구성
        all_choices_texts = ["Q: " + question + " A: " + row[col] for col in self.choice_cols]
        correct_choice_idx = self.choice_cols.index(correct_answer_col)

        # 프로세서가 토큰화 및 이미지 처리를 수행
        # padding='max_length'와 truncation=True를 추가하여 모든 텍스트를 동일한 길이로 만듭니다.
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

def train():
    """VQA를 위한 대조 학습을 사용하여 CLIP 모델을 파인튜닝합니다."""
    # 1. 설정
    print("학습 설정을 시작합니다...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # 2. 모델 및 프로세서 로드
    print(f"모델 로드 중: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 3. 데이터 로드
    print(f"데이터 로드 중: {TRAIN_CSV_PATH}")
    if not os.path.exists(TRAIN_CSV_PATH):
        print(f"오류: {TRAIN_CSV_PATH} 에서 학습 데이터를 찾을 수 없습니다.")
        return
        
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    dataset = VQADataset(df=train_df, processor=processor, image_dir=IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. 옵티마이저
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 5. 학습 루프
    print("학습을 시작합니다...")
    model.train() # 모델을 학습 모드로 설정

    for epoch in range(EPOCHS):
        print(f"--- 에포크 {epoch+1}/{EPOCHS} ---")
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"에포크 {epoch+1} 배치"):
            # 배치를 장치로 이동
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('label') # Shape: (batch_size,)

            # 입력 데이터 형태 변경
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            pixel_values = batch['pixel_values']
            
            batch_size = pixel_values.shape[0]
            num_choices = input_ids.shape[1]
            
            # 텍스트 입력의 형태를 (batch_size, num_choices, seq_len) -> (batch_size * num_choices, seq_len)으로 변경
            input_ids = input_ids.view(batch_size * num_choices, -1)
            attention_mask = attention_mask.view(batch_size * num_choices, -1)
            
            # 순전파
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
            
            # 각 이미지에 해당하는 4개의 선택지에 대한 로짓만 추출
            logits = outputs.logits_per_image
            
            # 각 이미지에 대한 올바른 로짓을 선택
            logits_for_loss = torch.zeros(batch_size, num_choices).to(device)
            for i in range(batch_size):
                logits_for_loss[i] = logits[i, i*num_choices : (i+1)*num_choices]

            # 손실 함수 계산
            loss = torch.nn.functional.cross_entropy(logits_for_loss, labels)

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"에포크 {epoch+1} 완료. 평균 손실: {avg_loss:.4f}")

    # 6. 모델 저장
    print(f"학습 완료. 모델을 {OUTPUT_MODEL_DIR}에 저장합니다.")
    model.save_pretrained(OUTPUT_MODEL_DIR)
    processor.save_pretrained(OUTPUT_MODEL_DIR)
    print("모델이 성공적으로 저장되었습니다.")

def generate_baseline_submission():
    """파인튜닝된 CLIP 모델로 test.csv에 대한 예측을 수행하고 baseline_submit.csv 파일을 생성합니다."""
    # 1. 장치 설정
    print("장치를 설정합니다...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 2. 파인튜닝된 CLIP 모델 및 프로세서 로드
    print(f"파인튜닝된 모델을 로드합니다: {FINETUNED_MODEL_PATH}...")
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"오류: {FINETUNED_MODEL_PATH} 에서 파인튜닝된 모델을 찾을 수 없습니다.")
        print("먼저 학습 스크립트를 실행해주세요.")
        return
        
    try:
        model = CLIPModel.from_pretrained(FINETUNED_MODEL_PATH).to(device)
        processor = CLIPProcessor.from_pretrained(FINETUNED_MODEL_PATH)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 3. 테스트 CSV 파일 로드
    print(f"테스트 데이터를 로드합니다: {TEST_CSV_PATH}...")
    if not os.path.exists(TEST_CSV_PATH):
        print(f"오류: {TEST_CSV_PATH} 파일을 찾을 수 없습니다.")
        return
        
    test_df = pd.read_csv(TEST_CSV_PATH)

    # 4. 정답 예측
    print("정답 예측을 시작합니다...")
    predictions = []
    choice_columns = ['A', 'B', 'C', 'D']
    
    model.eval() # 모델을 평가 모드로 설정

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="테스트 이미지 처리 중"):
        try:
            question = row.get('question', row.get('Question', ''))
            if not question:
                print(f"행 {row.get('ID', '?')}에 'question' 열이 없습니다. 건너뜁니다.")
                predictions.append("?")
                continue

            img_path = os.path.join(IMAGE_DIR, os.path.basename(row['img_path']))
            image = Image.open(img_path).convert("RGB")
            
            choices_text = [question + " " + str(row[col]) for col in choice_columns]
            inputs = processor(text=choices_text, images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            best_choice_idx = logits_per_image.softmax(dim=1).argmax().item()
            predicted_answer = choice_columns[best_choice_idx]
            predictions.append(predicted_answer)

        except Exception as e:
            print(f"행 {row['ID']} 처리 중 오류 발생: {e}")
            predictions.append("?")

    # 5. 제출 파일 생성
    print(f"\n제출 파일을 생성합니다: {SUBMISSION_CSV_PATH}...")
    # sample_submission.csv의 ID 순서를 따르기 위해 원본 test_df의 ID를 사용합니다.
    submission_df = pd.DataFrame({'ID': test_df['ID'], 'answer': predictions})
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)

    print(f"\n{SUBMISSION_CSV_PATH} 파일이 성공적으로 생성되었습니다.")
    print("--- 제출 파일 내용 (상위 5개) ---")
    print(submission_df.head())

if __name__ == "__main__":
    train()
    print("===================학습완료====================")
    print("파인튜닝된 모델로 baseline 제출 파일을 생성합니다...")
    generate_baseline_submission()