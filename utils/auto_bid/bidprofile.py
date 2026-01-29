import numpy as np



class Generator:
    def __init__(self):
        self.nMgc = 0
        self.nMg = 0

        self.dRurq1 = 0
        self.dRur1 = 0
        self.dRdr1 = 0

        self.dRurq2 = 0
        self.dRur2 = 0
        self.dRdr2 = 0

        self.dRurq3 = 0
        self.dRur3 = 0
        self.dRdr3 = 0

        self.dRurq4 = 0
        self.dRur4 = 0
        self.dRdr4 = 0


class Profile:
    ARRAY_SIZE_HOUR = 34
    ARRAY_SIZE_MINUTE = 60

    def __init__(self, avail, gen: Generator):
        self.avail = np.array(avail, dtype=float)  # 34개 시간 구간별 입찰량(1차원)
        self.gen = gen
        self.dAvail = np.zeros((self.ARRAY_SIZE_HOUR, self.ARRAY_SIZE_MINUTE), dtype=float)
        self.nProfileResult = np.zeros(self.ARRAY_SIZE_HOUR, dtype=int)
        self._init_avail()

    def _init_avail(self):
        for i in range(self.ARRAY_SIZE_HOUR):
            v = self.avail[i] if self.avail[i] >= 0 else 0
            self.dAvail[i, :] = v

    def calc_profile(self):
        # Forward Profile
        for i in range(self.ARRAY_SIZE_HOUR):
            for j in range(self.ARRAY_SIZE_MINUTE):
                if i > 0 and j == 0:
                    if self.dAvail[i-1, 59] >= 0 and self.dAvail[i-1, 59] < self.dAvail[i, 0]:
                        dRurTemp = self.get_target_rur(self.dAvail[i-1, 59])
                        dAvailTemp = self.dAvail[i-1, 59] + dRurTemp
                        self.dAvail[i, 0] = min(dAvailTemp, self.dAvail[i, 0])
                elif j > 0:
                    dRurTemp = self.get_target_rur(self.dAvail[i, j-1])
                    dAvailTemp = self.dAvail[i, j-1] + dRurTemp
                    self.dAvail[i, j] = min(dAvailTemp, self.dAvail[i, j])

        # Backward Profile
        for i in reversed(range(self.ARRAY_SIZE_HOUR)):
            for j in reversed(range(self.ARRAY_SIZE_MINUTE)):
                if i > 0 and j == 0:
                    if self.dAvail[i, 0] >= 0 and self.dAvail[i-1, 59] > self.dAvail[i, 0]:
                        dRdrTemp = self.get_target_rdr(self.dAvail[i, 0])
                        dAvailTemp = self.dAvail[i, 0] + dRdrTemp
                        self.dAvail[i-1, 59] = min(dAvailTemp, self.dAvail[i-1, 59])
                elif j > 0:
                    if self.dAvail[i, j] >= 0:
                        dRdrTemp = self.get_target_rdr(self.dAvail[i, j])
                        dAvailTemp = self.dAvail[i, j] + dRdrTemp
                        self.dAvail[i, j-1] = min(dAvailTemp, self.dAvail[i, j-1])

        # 1시간 평균값 계산 (소수점 절삭)
        for i in range(self.ARRAY_SIZE_HOUR):
            dCalcTemp = self.dAvail[i, :].sum()
            self.nProfileResult[i] = int(np.floor(dCalcTemp / self.ARRAY_SIZE_MINUTE))
            # 원 입력값이 음수이면 그대로 사용
            if self.avail[i] < 0:
                self.nProfileResult[i] = int(self.avail[i])
        return self.nProfileResult.tolist()

    def get_target_rur(self, dAvail):
        g = self.gen
        if dAvail < g.dRurq1:
            return g.dRur1
        elif dAvail < g.dRurq2:
            return g.dRur2
        elif dAvail < g.dRurq3:
            return g.dRur3
        else:
            return g.dRur4

    def get_target_rdr(self, dAvail):
        g = self.gen
        if dAvail < g.dRurq1:
            return g.dRdr1
        elif dAvail < g.dRurq2:
            return g.dRdr2
        elif dAvail < g.dRurq3:
            return g.dRdr3
        else:
            return g.dRdr4

class ReBidProfile:
    ARRAY_SIZE_HOUR = 34
    ARRAY_SIZE_MINUTE = 60

    def __init__(self, arrReAvail, arrBeforeProfile, gen: Generator):
        # arrReAvail: [ [시각1, 시각2, ...], [입찰량1, 입찰량2, ...] ] (2차원)
        self.arrReAvail = arrReAvail
        self.arrBeforeProfile = arrBeforeProfile
        self.gen = gen
        self.dAvail = np.zeros((self.ARRAY_SIZE_HOUR, self.ARRAY_SIZE_MINUTE), dtype=float)
        self.nProfileResult = np.zeros(self.ARRAY_SIZE_HOUR, dtype=int)
        self.arrPumpingTemp = np.array(arrBeforeProfile, dtype=float)

    def init_array(self):
        # 기존 profile로 초기화
        for i in range(self.ARRAY_SIZE_HOUR):
            self.dAvail[i, :] = self.arrBeforeProfile[i]

        # 변경입찰량 반영 (시점 이후 덮어쓰기)
        for idx, str_time in enumerate(self.arrReAvail[0]):
            nAvailTemp = float(self.arrReAvail[1][idx])
            col, row = get_profile_point(str_time)
            for i in range(col, self.ARRAY_SIZE_HOUR):
                for j in range(row if i == col else 0, self.ARRAY_SIZE_MINUTE):
                    self.dAvail[i, j] = max(nAvailTemp, 0)
            # arrPumpingTemp는 1시간 단위만 반영
            for i in range(col, self.ARRAY_SIZE_HOUR):
                self.arrPumpingTemp[i] = nAvailTemp

    def calc_profile(self):
        self.init_array()

        # Forward Profile
        for i in range(self.ARRAY_SIZE_HOUR):
            for j in range(self.ARRAY_SIZE_MINUTE):
                if i > 0 and j == 0:
                    if self.dAvail[i-1, 59] >= 0 and self.dAvail[i-1, 59] < self.dAvail[i, 0]:
                        dRurTemp = self.get_target_rur(self.dAvail[i-1, 59])
                        dAvailTemp = self.dAvail[i-1, 59] + dRurTemp
                        self.dAvail[i, 0] = min(dAvailTemp, self.dAvail[i, 0])
                elif j > 0:
                    dRurTemp = self.get_target_rur(self.dAvail[i, j-1])
                    dAvailTemp = self.dAvail[i, j-1] + dRurTemp
                    self.dAvail[i, j] = min(dAvailTemp, self.dAvail[i, j])

        # Backward Profile
        for i in reversed(range(self.ARRAY_SIZE_HOUR)):
            for j in reversed(range(self.ARRAY_SIZE_MINUTE)):
                if i > 0 and j == 0:
                    if self.dAvail[i, 0] >= 0 and self.dAvail[i-1, 59] > self.dAvail[i, 0]:
                        dRdrTemp = self.get_target_rdr(self.dAvail[i, 0])
                        dAvailTemp = self.dAvail[i, 0] + dRdrTemp
                        self.dAvail[i-1, 59] = min(dAvailTemp, self.dAvail[i-1, 59])
                elif j > 0:
                    if self.dAvail[i, j] >= 0:
                        dRdrTemp = self.get_target_rdr(self.dAvail[i, j])
                        dAvailTemp = self.dAvail[i, j] + dRdrTemp
                        self.dAvail[i, j-1] = min(dAvailTemp, self.dAvail[i, j-1])

        # 1시간 평균값
        for i in range(self.ARRAY_SIZE_HOUR):
            dCalcTemp = self.dAvail[i, :].sum()
            self.nProfileResult[i] = int(np.floor(dCalcTemp / self.ARRAY_SIZE_MINUTE))
            # 음수면 그대로 반영
            if self.arrPumpingTemp[i] < 0:
                self.nProfileResult[i] = int(self.arrPumpingTemp[i])
        return self.nProfileResult.tolist()

    def get_target_rur(self, dAvail):
        g = self.gen
        if dAvail < g.dRurq1:
            return g.dRur1
        elif dAvail < g.dRurq2:
            return g.dRur2
        elif dAvail < g.dRurq3:
            return g.dRur3
        else:
            return g.dRur4

    def get_target_rdr(self, dAvail):
        g = self.gen
        if dAvail < g.dRurq1:
            return g.dRdr1
        elif dAvail < g.dRurq2:
            return g.dRdr2
        elif dAvail < g.dRurq3:
            return g.dRdr3
        else:
            return g.dRdr4



def get_profile_point(str_time):
    """
    str_time: ex) '-1800', '*0901', '+0130'
    return: (col, row)
    """
    if not str_time or len(str_time) < 5:
        return 999, 0
    str_day = str_time[0]
    str_hour = int(str_time[1:3])
    str_minute = int(str_time[3:5])
    if str_day == "-":
        if str_minute == 0:
            if str_hour == 18:
                return 0, 0
            else:
                return str_hour - 19, 59
        elif str_minute == 1:
            if str_hour == 18:
                return 0, 0
            else:
                return str_hour - 18, 0
        else:
            return str_hour - 18, str_minute - 1
    elif str_day == "*":
        if str_minute == 0:
            return str_hour + 5, 59
        else:
            return str_hour + 6, str_minute - 1
    elif str_day == "+":
        if str_minute == 0:
            return str_hour + 29, 59
        else:
            return str_hour + 30, str_minute - 1
    else:
        return 999, 0

