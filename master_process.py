import pandas as pd
import random
import os
from tqdm import tqdm

# 이 스크립트가 위치한 디렉터리
CWD = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER_NAME = "image"

def convert_to_mcq(source_name, output_name):
    """질문-답변 형식의 CSV를 객관식(MCQ) 형식으로 변환합니다."""
    source_path = os.path.join(CWD, source_name)
    output_path = os.path.join(CWD, output_name)
    print(f"--- 객관식 변환 시작: {source_name} -> {output_name} ---")
    
    try:
        df = pd.read_csv(source_path)
        df.dropna(subset=['answer'], inplace=True)
        answer_pool = df['answer'].astype(str).unique().tolist()
        
        new_data = []
        choices_cols = ['A', 'B', 'C', 'D']

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {source_name}"):
            correct_answer = str(row['answer'])
            possible_distractors = [ans for ans in answer_pool if ans != correct_answer]
            num_distractors = min(3, len(possible_distractors))
            distractors = random.sample(possible_distractors, num_distractors)
            options = distractors + [correct_answer]
            while len(options) < 4:
                options.append("(내용 없음)")
            random.shuffle(options)
            correct_letter = choices_cols[options.index(correct_answer)]
            
            new_row = {
                'ID': row['ID'],
                'img_path': f'./{IMAGE_FOLDER_NAME}/{row["image_id"]}.jpg',
                'Question': row['question'],
                'A': options[0], 'B': options[1], 'C': options[2], 'D': options[3],
                'answer': correct_letter
            }
            new_data.append(new_row)

        df_new = pd.DataFrame(new_data)
        df_new.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"성공적으로 {output_path} 파일을 생성했습니다.")
        return True
    except FileNotFoundError:
        print(f"오류: 소스 파일 '{source_name}'을(를) 찾을 수 없습니다. 파일이 스크립트와 동일한 디렉토리에 있는지 확인하세요.")
        return False
    except Exception as e:
        print(f"오류 발생 ({source_name}): {e}")
        return False

def filter_yes_no_answers(source_name, output_name):
    """지정된 CSV 파일에서 'yes'/'no' 정답을 필터링하여 새 파일에 저장합니다."""
    source_path = os.path.join(CWD, source_name)
    output_path = os.path.join(CWD, output_name)
    print(f"--- Yes/No 필터링 시작: {source_name} -> {output_name} ---")
    
    try:
        df = pd.read_csv(source_path)
        initial_rows = len(df)
        print(f"초기 행의 수: {initial_rows}")

        def is_yes_no_answer(row):
            try:
                return str(row[row['answer']]).lower().strip() in ['yes', 'no']
            except KeyError:
                return False

        rows_to_keep = ~df.apply(is_yes_no_answer, axis=1)
        df_filtered = df[rows_to_keep]
        final_rows = len(df_filtered)
        
        print(f"필터링 후 행의 수: {final_rows} (제거된 행: {initial_rows - final_rows})")
        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"성공적으로 {output_path} 파일을 생성했습니다.")
        return True
    except Exception as e:
        print(f"오류 발생 ({source_name}): {e}")
        return False

if __name__ == "__main__":
    # --- 전체 프로세스 실행 ---
    print("데이터 준비 전체 프로세스를 시작합니다.")

    # 입력 파일 및 중간/최종 파일 이름 정의
    input_file = "train.csv"
    mcq_file = "train_mcq.csv"
    final_output_file = "train_final.csv"

    # 1. train.csv를 MCQ 형식으로 변환하여 train_mcq.csv로 저장
    if convert_to_mcq(input_file, mcq_file):
        # 2. 변환된 파일에서 Yes/No 답변을 필터링하여 train_final.csv로 저장
        if filter_yes_no_answers(mcq_file, final_output_file):
            print(f"\n모든 작업이 완료되었습니다. 최종 파일: {final_output_file}")
        else:
            print("\nYes/No 필터링 단계에서 오류가 발생하여 프로세스를 중단합니다.")
    else:
        print("\n객관식 변환 단계에서 오류가 발생하여 프로세스를 중단합니다.")