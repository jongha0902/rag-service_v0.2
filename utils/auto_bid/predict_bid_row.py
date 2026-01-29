# predict_bid_row.py
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import bidprofile
import bidvalidation   

scaler = joblib.load("data/auto_bid_data/scaler.pkl")
encoders = joblib.load("data/auto_bid_data/encoders.pkl")


# ==== 1. 온도기준 CSV 전처리 (콤마, float 등) ====
def clean_numeric(x):
    if isinstance(x, str):
        return float(x.replace(",", ""))
    return float(x)

gf_min  = 262
agc_min  = 262
a_min = 0
tempf = "data/auto_bid_data/복합발전기 온도관리.csv"
temp_df = pd.read_csv(tempf, encoding="utf-8")
num_cols = ["GT#1", "GT#2", "ST", "합계", "허용하한", "허용상한", "온도"]
for col in num_cols:
    temp_df[col] = temp_df[col].apply(clean_numeric)
temptable = temp_df.to_dict(orient="records")

# ==== 2. 모델/스케일러/인코더 로드 ====
feature_cols = [
    "BID_QT", "PBID_QT", "FUEL_QT", "TA_QT", "PTA_QT", "CONST_QT",
    "A_MAX", "A_MIN", "GEN_CHG", "GF_MAX", "GF_MIN", "AGC_MIN", "AGC_MAX", "ECR",
    "EW_CONST", "TEMPER",
    "CONST_TYPE", "CONST_REASON", "OPER_BIT", "AGC_FLG", "GF_FLG", "BSF_FLG", "GT_SEQ"
]



class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

model = MLP(len(feature_cols))
model.load_state_dict(torch.load("data/auto_bid_data/mlp_model.pth"))
model.eval()

# ==== 3. 예측 row 생성 ====
def make_predict_row(temp, OPER_BIT, temptable , TRADE_YMDH=None, GEN_CD=None ):
    global gf_min 
    global agc_min 
    global a_min 

    row = next((r for r in temptable if int(r["온도"]) == int(temp)), None)
    if row is None:
        raise ValueError(f"{temp}도에 대한 기준 데이터가 없습니다.")
    if OPER_BIT == "11":
        GT1, GT2, ST, 합계, GT_SEQ = row["GT#1"], row["GT#2"], row["ST"], row["합계"], "12"
    elif OPER_BIT == "10":
        GT1, GT2, ST, 합계, GT_SEQ = row["GT#1"], 0, int(row["ST"]/2), row["GT#1"]+int(row["ST"]/2), "1"
    elif OPER_BIT == "01":
        GT1, GT2, ST, 합계, GT_SEQ = 0, row["GT#2"], int(row["ST"]/2), row["GT#2"]+int(row["ST"]/2), "2"
    elif OPER_BIT == "0":
        GT1, GT2, ST, 합계, GT_SEQ = 0, 0, 0, 0, "0"        
    else:
        raise ValueError(f"알 수 없는 운전정보: {OPER_BIT}")
    predict = {
        "BID_QT": 합계,
        "PBID_QT": 합계,
        "FUEL_QT": 합계,
        "TA_QT": 합계,
        "PTA_QT": 합계,
        "CONST_QT": 0,
        "CONST_TYPE": "0",
        "CONST_REASON": "0",
        "OPER_BIT": OPER_BIT,
        "GT_SEQ": GT_SEQ,
        "A_MAX": 합계,
        "A_MIN": 0,
        "GEN_CHG": 1.015 ,
        "GF_MAX": 합계,
        "GF_MIN": gf_min,
        "AGC_MIN": agc_min,
        "AGC_MAX": 합계,
        "TEMPER": temp,
        "AGC_FLG": "1",
        "GF_FLG": "1",
        "BSF_FLG": "0",
        "ECR": 0,
        "EW_CONST": 0
    }
    return predict


# ==== 4. 판정(규칙 + 딥러닝) ====
def safe_transform(encoder, series):
    known_classes = set(encoder.classes_)
    new_vals = series.astype(str).values
    transformed = []
    for v in new_vals:
        if v in known_classes:
            transformed.append(encoder.transform([v])[0])
        else:
            transformed.append(-1)
    return transformed


def make_predict_rows(user_df):
    result_list = []
    pred_rows = []   # 각 기준값 dict를 여기에 쌓음
    avail = []       # BID_QT만 따로 추출(1차원 리스트)
    tavail = []       # TA만 따로 추출(1차원 리스트)

    for idx, row in user_df.iterrows():
        temp = row['TEMPER']
        oper_bit = str(row['OPER_BIT'])
        trade_ymdh = row['TRADE_YMDH']
        gen_cd = row['GEN_CD']

        # 기준 데이터 생성
        pred_row = make_predict_row(temp, oper_bit, temptable, trade_ymdh, gen_cd)
        pred_rows.append(pred_row)  # 기준값(딕셔너리) 추가
        avail.append(float(pred_row['BID_QT']))  # Profile에 쓸 BID_QT 값 쌓기
        tavail.append(float(pred_row['TA_QT']))  # Profile에 쓸 BID_QT 값 쌓기
        # 비교용 결과 생성
        #item = {'TRADE_YMDH': trade_ymdh, 'GEN_CD': gen_cd}
        #for col in ['BID_QT', 'PBID_QT', 'FUEL_QT', 'TA_QT', 'PTA_QT']:
        #    user_v = float(row[col])
        #    std_v = float(pred_row[col])
        #    diff = abs(user_v - std_v) / (std_v + 1e-9)
        #    item[col+'_적합여부'] = '적합' if diff < 0.05 else f"부적합({diff:.1%})"
        #    item[col+'_기준값'] = std_v
        #    item[col+'_차이'] = user_v - std_v

        #result_list.append(item)

    # --- Profile 계산 ---
    gen = bidprofile.Generator()
    gen.dRurq1 = 1052; gen.dRur1 = 55; gen.dRdr1 = 55
    # 필요하면 gen의 나머지 값도 할당

    # avail은 34개 시간구간 BID_QT값이어야 합니다. (필요에 따라 체크)
    

    rprofile = bidprofile.Profile(avail, gen)
    profile_result = rprofile.calc_profile()

    tprofile = bidprofile.Profile(avail, gen)
    tprofile_result = tprofile.calc_profile()

    for i, row in enumerate(pred_rows):
        row['PBID_QT'] = profile_result[i]
        row['FUEL_QT'] = profile_result[i]
        row['PTA_QT'] = tprofile_result[i]



    # 필요하면 result_list나 profile_result를 반환
    return pred_rows



def check_and_predict(new_df, encoders, scaler, model, feature_cols, temptable):
    """
    new_df: 평가할 유저 DataFrame
    pred_rows: 시스템 기준값 dict 리스트 (make_predict_rows에서 생성)
    """
    # 1. 비교 기준값 생성
    pred_rows = make_predict_rows(new_df)

    print("new_df 행 개수:", len(new_df))
    print("pred_rows 개수:", len(pred_rows))
    print("입력:", new_df)
    print("예측", pred_rows)

    # 2. 규칙 위반 체크
    new_df = new_df.copy()
    for col in ['CONST_TYPE', 'CONST_REASON', 'OPER_BIT', 'AGC_FLG', 'GF_FLG', 'BSF_FLG', 'GT_SEQ', 'GEN_CHG']:
        new_df[col] = new_df[col].astype(str)
    new_df['violations'] = new_df.apply(bidvalidation.validate_bid, axis=1)
    new_df['rule_pass'] = new_df['violations'].apply(lambda x: len(x) == 0)

    # 3. 범주형 인코딩
    for col in encoders.keys():
        new_df[col] = safe_transform(encoders[col], new_df[col])

    # 4. 딥러닝 판정
    X_new = new_df[feature_cols].values
    X_new_scaled = scaler.transform(X_new)
    X_tensor_new = torch.tensor(X_new_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor_new)
        pred = torch.argmax(logits, dim=1).numpy()
    new_df['mlp_result'] = pred

    # 5. **new_df와 pred_rows 비교** 및 최종적합성 판정
    compare_cols = ["BID_QT", "PBID_QT", "FUEL_QT", "TA_QT", "PTA_QT"]
    results = []
    for i, (idx, user_row) in enumerate(new_df.iterrows()):
        pred_row = pred_rows[i]
        compare_info = {}
        value_pass = True  # 값기반 적합여부 플래그

        for col in compare_cols:
            user_val = float(user_row[col])
            std_val = float(pred_row[col])
            diff = user_val - std_val
            rel_err = abs(user_val - std_val) / (abs(std_val) + 1e-8)
            적합 = '적합' if rel_err < 0.05 else f'부적합({rel_err:.1%})'
            compare_info[f'{col}_기준값'] = std_val
            compare_info[f'{col}_차이'] = diff
            compare_info[f'{col}_적합여부'] = 적합
            if '부적합' in 적합:
                value_pass = False

        # 판정 우선순위: 규칙 위반 > 값 부적합 > 딥러닝 부적합 > 적합
        if not user_row['rule_pass']:
            final_result = '부적합(규칙위반)'
        elif not value_pass:
            final_result = '부적합(값오차)'
        elif user_row['mlp_result'] != 1:
            final_result = '부적합(딥러닝)'
        else:
            final_result = '적합'

        result_row = {
            'TRADE_YMDH': user_row.get('TRADE_YMDH', ''),
            'GEN_CD': user_row.get('GEN_CD', ''),
            'rule_pass': user_row['rule_pass'],
            'violations': user_row['violations'],
            'mlp_result': '적합' if user_row['mlp_result'] == 1 else '부적합',
            '최종적합성': final_result
        }
        result_row.update(compare_info)
        results.append(result_row)

    return pd.DataFrame(results)



