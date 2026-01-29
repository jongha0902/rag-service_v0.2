import requests
import certifi
import sqlite3
from datetime import datetime, timedelta


# API 파라미터 예시 (실제 코드에서는 외부에서 주어짐)
# params = {
#     'regId': '11H20401',
#     'tmFc': '202506080600'
# }
# item = {...}  # getMidTa API로부터 받아온 JSON 응답에서 'item'

# 기준일 계산


# DB 연결 및 테이블 생성

def create_db():
    conn = sqlite3.connect("D:/eGovFrame-4.0.0/db/brms.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mid_term_temperature (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reg_id TEXT NOT NULL,
        tm_fc TEXT NOT NULL,
        forecast_date TEXT NOT NULL,
        ta_min INTEGER,
        ta_max INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(reg_id, forecast_date)
    )
    """)
    return conn, cursor 
def get_settlement_timestamp():
    now = datetime.now()

    if now.hour < 6:
        target_date = now - timedelta(days=1)
        time_suffix = '1800'
    elif now.hour < 18:
        target_date = now
        time_suffix = '0600'
    else:
        target_date = now
        time_suffix = '1800'

    return target_date.strftime('%Y%m%d') + time_suffix

def get_mid_term_temperature(reg_id):
    conn, cursor =  create_db()
    url = 'http://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa'
    params = {
        'serviceKey': '6QCeObXVx0GK0r3PYtUzTF4BQcIA2MM65+MtplS2lERt4CW8wnwdVKfoLqXYpVNuii6sr03vTYzoQY77VFDB9w==',  # 실제 키
        'pageNo': '1',
        'numOfRows': '10',
        'dataType': 'JSON',
        'regId': reg_id,
        'tmFc': get_settlement_timestamp()
    }
    base_date = datetime.strptime(params['tmFc'], "%Y%m%d%H%M")
    response = requests.get(url, params=params, verify=certifi.where())
    data = response.json()

    item = data['response']['body']['items']['item'][0]

    reg_id = params['regId']
    tm_fc = params['tmFc']

    for day_offset in range(3, 11):
        forecast_date = (base_date + timedelta(days=day_offset)).strftime("%Y%m%d")
        ta_min = item.get(f'taMin{day_offset}')
        ta_max = item.get(f'taMax{day_offset}')

        # 중복 확인 후 INSERT
        cursor.execute("""
            SELECT 1 FROM mid_term_temperature
            WHERE reg_id = ? AND forecast_date = ?
        """, (reg_id, forecast_date))

        if cursor.fetchone() is None:
            cursor.execute("""
                INSERT INTO mid_term_temperature (reg_id, tm_fc, forecast_date, ta_min, ta_max)
                VALUES (?, ?, ?, ?, ?)
            """, (reg_id, tm_fc, forecast_date, ta_min, ta_max))

    conn.commit()
    conn.close()


if __name__ == '__main__':
    get_mid_term_temperature("11H20401")
