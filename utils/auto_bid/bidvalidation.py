#bidvalidation.py
def validate_bid(row):
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

    def to_float(x):
        try: return float(x)
        except: return 0

    # GEN_CHG 1 이상 2 미만 필수
    if not (1 <= to_float(row['GEN_CHG']) < 2):
        violations.append('GEN_CHG는 1이상 2미만이어야 함')

    if (1 == to_float(row['GEN_CHG']) and pbid_qt >0 ):
        violations.append(f"프로파일된 입찰값이 {pbid_qt}이상인데.. 발전단 전환비가 1입니다. 수정필요합니다." )    

    if to_float(row['CONST_QT']) > 0:
        if str(row['CONST_TYPE']) == "0":
            violations.append('CONST_TYPE must not be 0 when CONST_QT > 0')
        if str(row['CONST_REASON']) == "0":
            violations.append('CONST_REASON must not be 0 when CONST_QT > 0')
    if str(row['OPER_BIT']) != "00" and to_float(row['BID_QT']) < 0:
        violations.append('BID_QT must be >= 0 when OPER_BIT != 00')
    
    # 조건별 유효성 검토
    if bid_qt > mgc:
        violations.append("BID_QT > MGC")
    if bid_qt < const_qt:
        violations.append("BID_QT < CONST_QT")
    if ta_qt > mgc:
        violations.append("TA_QT > MGC")
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

    return violations


