import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import os

# --- 설정 ---
FINETUNED_MODEL_PATH = "./clip_finetuned"
TEST_CSV_PATH = "train_final.csv"
IMAGE_DIR = "train/"
SUBMISSION_CSV_PATH = "baseline_submit_tuned.csv"

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
    generate_baseline_submission()