import sqlite3
from datetime import datetime, timedelta

def get_temperature_summary(target_date: str, reg_id: str = '11H20401', db_path="D:/eGovFrame-4.0.0/db/brms.db") -> dict:
    next_date = (datetime.strptime(target_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
    WITH latest_dday AS (
      SELECT ta_max, ta_min
      FROM mid_term_temperature
      WHERE reg_id = ?
        AND forecast_date = ?
        AND tm_fc = (
          SELECT MAX(tm_fc)
          FROM mid_term_temperature
          WHERE reg_id = ?
            AND forecast_date = ?
        )
    ),
    latest_nday AS (
      SELECT ta_min
      FROM mid_term_temperature
      WHERE reg_id = ?
        AND forecast_date = ?
        AND tm_fc = (
          SELECT MAX(tm_fc)
          FROM mid_term_temperature
          WHERE reg_id = ?
            AND forecast_date = ?
        )
    )
    SELECT
      d.ta_max AS dmax,
      d.ta_min AS dmin,
      n.ta_min AS nmin
    FROM latest_dday d, latest_nday n;
    """

    params = (
        reg_id, target_date,
        reg_id, target_date,
        reg_id, next_date,
        reg_id, next_date
    )

    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()

    if row:
        return {'dmax': row[0], 'dmin': row[1], 'nmin': row[2]}
    else:
        return {'dmax': None, 'dmin': None, 'nmin': None}

# 예시 사용
if __name__ == '__main__':
  result = get_temperature_summary("20250612", "11H20401")
  print(result)  # {'dmax': 25, 'dmin': 20, 'nmin': 19}
