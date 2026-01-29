import pandas as pd
import requests
import logging
import json
import predict_bid_row 
import joblib
import torch
from collections import Counter
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma:7b"
API_URL = "http://localhost:8080/etsapi/api/etb112rtlist"
API_HIST_URL = "http://localhost:8080/etsapi/api/etbHistlist"

def bid112er_column_desc():
    return {
        "TRADE_YMDH": "거래일자시간",
        "GEN_CD": "발전기코드",
        "REVISION": "입찰차수",
        "BID_QT": "공급가능용량",
        "PBID_QT": "프로파일된 공급가능용량",
        "FUEL_QT": "연료량",
        "TA_QT": "온도별 공급용량",
        "PTA_QT": "프로파일된 온도별 공급용량",
        "CONST_QT": "제약량",
        "CONST_TYPE": "제약유형",
        "CONST_REASON": "제약사유",
        "OPER_BIT": "운전표시",
        "AGC_FLG": "AGC여부",
        "GF_FLG": "GF여부",
        "BSF_FLG": "BSF여부",
        "TEMPER": "온도",
        "A_MAX": "최대발전용량",
        "A_MIN": "최소발전용량",
        "GEN_CHG": "발전단전환비",
        "GF_MAX": "GF상한",
        "GF_MIN": "GF하한",
        "AGC_MIN": "AGC하한",
        "AGC_MAX": "AGC상한",
        "ECR": "비상대기예비력",
        "EW_CONST": "환경제약",
        "GT_SEQ": "GT기동순위",
        "SUB_FUEL_USE_YN": "대체연료사용여부",
        "WRT_ID": "작성자",
        "WRT_YMDS": "작성일시",
        'A_MGMIN': '출력하한치'
    }

def get_bid_data(vtrade_ymd, vgen_cd, vrevision="0"):
    try:
        payload = {
            "tradeYmd": vtrade_ymd,
            "genCd": vgen_cd,
            "revision": vrevision
        }
        res = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=100)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logging.error(f"입찰 API 호출 실패: {e}")
        return None
    
def get_bidhist_data(vtrade_ymd, vgen_cd, vrevision="0"):
    try:
        payload = {
            "tradeYmd": vtrade_ymd,
            "genCd": vgen_cd,
            "revision": vrevision
        }
        res = requests.post(API_HIST_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=100)
        res.raise_for_status()
       
        

        return res.json()
    
    except Exception as e:
        logging.error(f"입찰 API 호출 실패: {e}")
        return None
def validate_bid_row(row):
    violations = []

    bid_qt = abs(row["BID_QT"])
    pbid_qt = abs(row["PBID_QT"])
    ta_qt = abs(row["TA_QT"]) if "TA_QT" in row else 0
    const_qt = row["CONST_QT"]
    a_max = row["A_MAX"]
    a_min = row["A_MIN"]
    mgc = row["A_MAX"]
    gen_chg = row["GEN_CHG"]
    gf_flg = str(row["GF_FLG"]).upper()
    gf_max = row["GF_MAX"]
    gf_min = row["GF_MIN"]
    agc_flg = str(row["AGC_FLG"]).upper()
    agc_max = row["AGC_MAX"]
    agc_min = row["AGC_MIN"]

    if bid_qt > mgc:
        violations.append("BID_QT > MGC")
    if bid_qt < const_qt:
        violations.append("BID_QT < CONST_QT")
    if ta_qt > mgc:
        violations.append("TA_QT > MGC")
    if gen_chg >= 2 or gen_chg < 1:
        violations.append("GEN_CHG out of bounds")
    if gen_chg <= 1  and bid_qt > 0:
        violations.append("입찰값이 있는 경우 발전단전환비율 1이하 일수 없습니다. ")

    if a_max > mgc:
        violations.append("A_MAX > MGC")
    if pbid_qt > a_max:
        violations.append("PBID_QT > A_MAX")
    if bid_qt < gf_max:
        violations.append("GF_MAX > BID_QT")
    if gf_max < agc_max:
        violations.append("AGC_MAX > GF_MAX")

    if gf_flg == "N" and (gf_max > 0 or gf_min > 0):
        violations.append("GF_FLG=N but GF_MIN/MAX > 0")
    if agc_flg == "N" and (agc_max > 0 or agc_min > 0):
        violations.append("AGC_FLG=N but AGC_MIN/MAX > 0")
    if gf_flg == "N" and agc_flg == "Y":
        violations.append("GF_FLG=N but AGC_FLG=Y")

    if gf_flg == "Y" and agc_flg == "N":
        if not (a_min <= gf_min <= gf_max <= a_max):
            violations.append("GF_MIN/MAX out of bounds")

    if gf_flg == "N" and agc_flg == "N":
        if a_min > a_max:
            violations.append("A_MIN > A_MAX")

    if gf_flg == "Y" and agc_flg == "Y":
        if not (a_min <= gf_min <= agc_min <= agc_max <= gf_max <= a_max):
            violations.append("GF/AGC MIN/MAX out of bounds")

    return "; ".join(violations)

def build_llm_prompt(dl_results2, bad_rows_info):
    total = len(dl_results2)
    bad_cnt = 0 if bad_rows_info.strip() == "없음" else bad_rows_info.count("행 |")
    good_cnt = total - bad_cnt
    good_pct = f"{(good_cnt/total*100):.1f}%"
    bad_pct = f"{(bad_cnt/total*100):.1f}%"

    return f"""📌 모든 출력은 반드시 한국어로만 작성하십시오. 영어 금지, 혼합문장 금지, 불필요한 설명 금지.

        아래는 전체 입찰적합성 평가결과입니다.

        ### ✅ 1. 총괄 판정
        - 전체 행 수: {total}
        - 적합: {good_cnt}건
        - 부적합: {bad_cnt}건
        - 적합비율: {good_pct}, 부적합비율: {bad_pct}
        - {'최종판정: 부적합' if bad_cnt > 0 else '최종판정: 적합'}

        ### ⚠ 2. 주요 부적합 내역
        {bad_rows_info}

        **위 기준에 따라 간결하게 실무 보고 형태로 작성하십시오.**
        """

def get_bad_rows_full(dl_results2):
    bad_rows = []
    for i, row in dl_results2.iterrows():
        bad_cols = []
        for col in ["BID_QT_적합여부","PBID_QT_적합여부","FUEL_QT_적합여부","TA_QT_적합여부","PTA_QT_적합여부"]:
            if str(row[col]).startswith("부적합"):
                bad_cols.append(f"{col.replace('_적합여부','')} {row[col]}")
        rule_viol = ""
        # 규칙 위반/딥러닝 판정 모두 부적합 포함
        if row.get("rule_pass") is False:
            rule_viol = f"[규칙 위반: {row.get('violations', '')}]"
        if row.get("mlp_result") == "부적합":
            rule_viol += f" [딥러닝 부적합]"
        if bad_cols or rule_viol:
            bad_rows.append(
                f"{i+1}번째 행 | 거래일자시간:{row.TRADE_YMDH} | "
                f"부적합: {', '.join(bad_cols) if bad_cols else ''} {rule_viol}".strip()
            )
    if not bad_rows:
        return "없음"
    return "\n".join(bad_rows)




def analyze_bid(prompt):
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=240)
        res.raise_for_status()
        return res.json().get("response", "[⚠️] 분석 실패")
    except Exception as e:
        return f"[❌] Ollama 요청 실패: {e}"

def rule_and_stat_check(datajson, histjson):
    df_new = pd.DataFrame(datajson)
    df_hist = pd.DataFrame(histjson)
    stat_cols = ['BID_QT', 'PBID_QT', 'FUEL_QT', 'TA_QT', 'A_MAX', 'A_MIN']
    stats = {col: (df_hist[col].astype(float).mean(), df_hist[col].astype(float).std()) for col in stat_cols}

    def check_stat_outlier(row, stats):
        reasons = []
        for col in stat_cols:
            value = float(row[col])
            mean, std = stats[col]
            if std > 0:
                z = abs((value - mean) / std)
                if z > 2.0:
                    reasons.append(f"{col} 이력평균(±2σ) 벗어남(z={z:.2f})")
        return "; ".join(reasons)

    results = []
    for idx, row in df_new.iterrows():
        violations = validate_bid_row(row)
        stat_outlier = check_stat_outlier(row, stats)
        if violations or stat_outlier:
            valid = "부적합"
            reason = "; ".join(filter(None, [violations, stat_outlier]))
        else:
            valid = "적합"
            reason = ""
        results.append({
            "TRADE_YMDH": row["TRADE_YMDH"],
            "적합성": valid,
            "사유": reason
        })
    return results


def analexec(vtrade_ymd = "20250708", vgen_cd = "7284", vrevision = "0"):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    datajson = get_bid_data(vtrade_ymd, vgen_cd, vrevision)
    #histjson = get_bidhist_data(vtrade_ymd, vgen_cd, vrevision)
    if not datajson:
        print("입찰 데이터가 없습니다.")
        exit

    df = pd.DataFrame(datajson)
    os.makedirs("bidhistory", exist_ok=True)
    logging.info(f"거래일자: {vtrade_ymd}, 발전기코드: {vgen_cd}, 입찰차수: {vrevision}")
    logging.info("입찰 데이터 수신 완료")

    descdf = bid112er_column_desc()
    scaler = joblib.load("data/auto_bid_data/scaler.pkl")
    encoders = joblib.load("data/auto_bid_data/encoders.pkl")
    model = predict_bid_row.MLP(len(predict_bid_row.feature_cols))
    model.load_state_dict(torch.load("data/auto_bid_data/mlp_model.pth"))
    model.eval()

    dl_results  = predict_bid_row.check_and_predict(df, encoders, scaler, model, predict_bid_row.feature_cols , predict_bid_row.temptable)
    dl_results2  = pd.concat([df[["TRADE_YMDH", "GEN_CD"]] , dl_results], axis=1)
    bad_rows_info = get_bad_rows_full(dl_results2)
    print("✅ CSV 저장 완료: 입찰적합성평가결과.csv")
    prompt = build_llm_prompt(dl_results2 , bad_rows_info ) 
    result = analyze_bid(prompt)

    
    print("\n📈 LLM 분석 결과:\n")
    print(result)
    return result

if __name__ == "__main__":
    analexec(vtrade_ymd = "20250721", vgen_cd = "7284", vrevision = "0")