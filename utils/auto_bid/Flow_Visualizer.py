import json
import sqlite3
import html
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import tkinter as tk
from tkinter import messagebox
import threading

from tmp_summary import get_temperature_summary
import MidFcst
import call_api
import re



_visual_root = None
_max_canvas_width = 1600
_max_canvas_height = 1000
_canvas = None
_positions = {}
target_date = datetime.now().strftime("%Y%m%d") 
tmpmid = None
genInfo="" 

def assign_node_positions(flow_data):
    positions = {}
    used = set()
    max_x = 0
    max_y = 0
    for node in flow_data["nodes"]:
        x = node.get("left", 100)
        y = node.get("top", 100)
        while (x, y) in used:
            y += 30
        used.add((x, y))
        positions[node["id"]] = (x + 100, y + 50)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    global _max_canvas_width, _max_canvas_height
    _max_canvas_width = max_x + 400
    _max_canvas_height = max_y + 400
    return positions


def update_node_status_safe(node_id, status, label=""):
    def _update():
        update_node_status(node_id, status, label)
    if _visual_root:
        _visual_root.after(0, _update)


def update_node_status(node_id, status, label=""):
    global _canvas
    color = {
        "success": "#a8e6cf",
        "fail": "#ff8b94",
        "warning": "#fff3b0"
    }.get(status, "#dddddd")

    if node_id not in _positions:
        print(f"⚠️ update_node_status: node_id '{node_id}' not found in _positions.")
        return

    x, y = _positions[node_id]
    _canvas.delete("node_" + node_id)
    _canvas.delete("label_" + node_id)
    _canvas.delete("status_" + node_id)

    _canvas.create_rectangle(x-60, y-30, x+60, y+30, fill=color, outline="black", width=2, tags="node_" + node_id)
    _canvas.create_text(x, y-12, text=label, tags="label_" + node_id, font=("Arial", 10, "bold"), width=110)
    icon = "✅" if status == "success" else "❌" if status == "fail" else "⚠️"
    _canvas.create_text(x, y+10, text=icon, tags="status_" + node_id, font=("Arial", 12))
    _canvas.update()


def init_visualizer(flow_data):
    global _visual_root, _canvas, _positions

    canvas_width = _max_canvas_width
    canvas_height = _max_canvas_height

    _visual_root = tk.Tk()
    _visual_root.title("Flow 실시간 실행 상태")
    _canvas = tk.Canvas(_visual_root, width=canvas_width, height=canvas_height, bg="white")
    _canvas.pack()

    _positions = assign_node_positions(flow_data)

    for node in flow_data["nodes"]:
        node_id = node["id"]
        label = node.get("ruleName", node_id)
        x, y = _positions[node_id]
        if len(label) > 14:
            label = label[:14] + "…"
        _canvas.create_rectangle(x-60, y-30, x+60, y+30, fill="#eeeeee", outline="black", width=2, tags="node_" + node_id)
        _canvas.create_text(x, y, text=label, tags="label_" + node_id, font=("Arial", 10), width=110)

    for conn in flow_data["connections"]:
        from_id = conn["from"]
        to_id = conn["to"]
        from_pos = _positions.get(from_id)
        to_pos = _positions.get(to_id)
        if from_pos and to_pos:
            x1, y1 = from_pos
            x2, y2 = to_pos
            if abs(x2 - x1) > 150:
                if x1 < x2:
                    x1 += 60
                    x2 -= 60
                else:
                    x1 -= 60
                    x2 += 60
                _canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, width=1.5)
            else:
                _canvas.create_line(x1, y1, x1 + 20, y1, x1 + 20, y2, x2, y2, arrow=tk.LAST, width=1.5)

    _visual_root.update()


def start_visualizer_thread(flow_data):
    def _start():
        global _visual_root
        if _visual_root is not None:
            try:
                _visual_root.destroy()  # 이전 시각화 윈도우 종료
                _visual_root = None
            except Exception as e:
                print(f"🔁 기존 시각화 종료 실패: {e}")

        init_visualizer(flow_data)
        _visual_root.mainloop()

    t = threading.Thread(target=_start, daemon=True)
    t.start()

def shutdown_visualizer():
    global _visual_root
    if _visual_root is not None:
        try:
            _visual_root.quit()       # 메인루프 종료 요청
            _visual_root.destroy()    # UI 제거
            _visual_root = None
            print("🧹 시각화 UI 정상 종료됨")
        except Exception as e:
            print(f"❌ 시각화 종료 실패: {e}")



def draw_execution_result(flow_data, results):
    node_map = {node["id"]: node for node in flow_data["nodes"]}
    result_map = {r["step"].get("nodeId", "") or r.get("nodeId", ""): r for r in results}

    canvas_width = 1000
    canvas_height = 300

    root = tk.Tk()
    root.title("Flow 실행 결과 시각화")
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()

    positions = {}
    x, y = 100, 150
    x_gap = 160

    for node in flow_data["nodes"]:
        node_id = node["id"]
        label = node.get("ruleName", node_id)
        status = result_map.get(node_id, {}).get("status", "none")

        color = {
            "success": "#a8e6cf",
            "fail": "#ff8b94",
            "warning": "#fff3b0"
        }.get(status, "#dddddd")

        canvas.create_rectangle(x, y-30, x+120, y+30, fill=color, outline="black", width=2)
        canvas.create_text(x+60, y, text=label + ("\n✅" if status == "success" else "\n❌" if status == "fail" else "\n⚠️" if status == "warning" else ""), font=("Arial", 10))
        positions[node_id] = (x+60, y)
        x += x_gap

    for conn in flow_data["connections"]:
        from_pos = positions.get(conn["from"])
        to_pos = positions.get(conn["to"])
        if from_pos and to_pos:
            canvas.create_line(from_pos[0]+60, from_pos[1], to_pos[0]-60, to_pos[1], arrow=tk.LAST)

    root.mainloop()



def list_flows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT FLOW_ID, CREATED_AT FROM WEBRULE_FLOW ORDER BY CREATED_AT DESC").fetchall()


def load_flow_by_id(db_path, flow_id):
    with sqlite3.connect(db_path) as conn:
        flow_row = conn.execute("SELECT FLOW_JSON FROM WEBRULE_FLOW WHERE FLOW_ID = ?", (flow_id,)).fetchone()
    if not flow_row:
        raise ValueError(f"FLOW_ID {flow_id} not found")
    return json.loads(flow_row[0])


def load_db_rules(db_path):
    with sqlite3.connect(db_path) as conn:
        rules = conn.execute("SELECT RULE_ID, RULE_NAME, ACTION_JSON FROM WEBRULES WHERE IS_ACTIVE = 'Y'").fetchall()
    return {
        str(rule_id): json.loads(html.unescape(action_json))
        for rule_id, _, action_json in rules
    }


def get_execution_sequence(flow):
    in_degree = defaultdict(int)
    graph = defaultdict(list)
    for conn in flow["connections"]:
        graph[conn["from"]].append(conn["to"])
        in_degree[conn["to"]] += 1

    queue = deque(n["id"] for n in flow["nodes"] if n["type"] == "start")
    visited = set()
    ordered_nodes = []

    while queue:
        node_id = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)
        ordered_nodes.append(node_id)
        for neighbor in graph[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] <= 0:
                queue.append(neighbor)

    return ordered_nodes

def detect_nexacro_message(driver, wait):
    message_checks = [
        {
            "xpaths": [
                "//div[@id='mainframe.VFrameSet.frameMain.msg.freemsg.form.grdMsg.body.gridrow_0.cell_0_0:text']"
            ],
            "button_xpath": "//div[@id='mainframe.VFrameSet.frameMain.msg.freemsg.form.btnOk']",
            "fail_keywords": ["오류", "Exception"]
        },
        {
           "xpaths": [
                "//div[contains(@id,'save.nochange') and contains(@id,'cell_0_0')]"
            ],
            "button_xpath": "//div[contains(@id,'save.nochange') and contains(@id,'btnOk')]",
            "fail_keywords": ["없습니다"]
        },
        {
           "xpaths": [
                "//div[contains(@id,'grdMsg')]"
            ],
            "button_xpath": "//div[contains(@id,'grdMsg') and contains(@id,'btnOk')]",
            "fail_keywords": ["계속 진행 하시겠습니까?"]
        }
    ]

    for check in message_checks:
        for xpath in check["xpaths"]:
            try:
                msg_el = driver.find_element(By.XPATH, xpath)
                message = msg_el.text.strip()
                if not message:
                    continue

                btn = wait.until(EC.element_to_be_clickable((By.XPATH, check["button_xpath"])))
                btn.click()

                status = "fail" if any(k in message for k in check["fail_keywords"]) else "warning"
                return {"status": status, "message": message}
            except Exception:
                print(f"Error detecting Nexacro message: {xpath}")
                print(Exception)
                continue  # 다른 XPath 후보로 넘어감

    return None


def process_step(driver, wait, actions, step , target_date, genInfo ):
    action_type = step.get("action")
    by = step.get("by", "xpath")
    selector = step.get("selector")
    label = step.get("label")
    try:
        if action_type == "click":
            if by == "id" and ":" in selector:
                element = driver.execute_script(f'return document.getElementById("{selector}");')
            elif by == "name" and ":" in selector:
                element = driver.execute_script(f'return document.getElementByName("{selector}");')
            else:
                by_type = By.ID if by == "id" else By.XPATH
                if not selector and not isinstance(label, str):
                    raise ValueError(f"Invalid step: {step}")
                xpath = selector if selector else f"//div[contains(text(), '{label}')]"
                element = wait.until(EC.element_to_be_clickable((by_type, xpath)))
            if element and element.is_displayed():
                actions.move_to_element(element).click().perform()

        elif action_type == "input":
            
            by_type = By.ID if by == "id" else By.XPATH
            if by=="id":
                by_type = By.ID 
            elif by=="name":
                by_type =By.NAME    
            else:
                by_type =By.XPATH        
            element = wait.until(EC.presence_of_element_located((by_type, selector)))
            if "calendaredit" in selector :
                if step.get("value") == "inputdt":
                     step["value"] = target_date 
                elif step.get("value") == "next_day":
                    step["value"] = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
                    target_date = step["value"]

            if "mskTemper1" in selector and tmpmid is not None :
                step["value"]  =  tmpmid["dmax"]
            if "mskTemper2" in selector and tmpmid is not None :
                step["value"]  =  tmpmid["dmin"]
            if "mskTemper4" in selector and tmpmid is not None :
                step["value"]  =  tmpmid["nmin"]

            element.click()
            element.clear()
            element.send_keys(step["value"])
            if step.get("enter"):
                element.send_keys(Keys.ENTER)
            if step.get("submit"):
                element.send_keys(Keys.RETURN)

        elif action_type == "check_message":
            result = detect_nexacro_message(driver, wait)
            if result:
                msg = result["message"]
                for rule in step.get("messages", []):
                    if rule["contains"] in msg:
                        return {
                            "status": rule.get("status", "warning"),
                            "step": step,
                            "message": msg,
                            "next": rule.get("next")
                        }
            return {
                "status": "success",
                "step": step,
                "message": "No message matched"
            }
        elif action_type == "checkstatus":
            by_type = By.ID if by == "id" else By.XPATH
            element = wait.until(EC.presence_of_element_located((by_type, selector)))
            text = element.text.strip()
            if step.get("value") in text:
                decision = step.get("next")
                return {
                    "status": "branch",
                    "step": step,
                    "message": f"staMagamFlg 판단 결과: {decision}",
                    "next": decision
                }
            return {
                "status": "success",
                "step": step,
                "message": "No message matched"
            }
        elif action_type == "gettmpmid":
            # 중기기온예보를 확인하는 작업 만            
            tmpmid = MidFcst.get_mid_term_temperature(step["region_id"])
        elif action_type == "tmpmid":
            # 중기기온예보를 확인하는 작업 만
            by_type = By.ID if by == "id" else By.XPATH
            tmpmid = get_temperature_summary(target_date, step["region_id"])

        elif action_type == "analbidinit":      
            vgen_cd = genInfo[:4]     # genInfo는 "7284 통영천연가스CC" 형태로 가정
            result = call_api.analexec(target_date, vgen_cd, "0")
            decision = step.get("next")
            judg = extract_judgement(result)
            print("\n📈 LLM 분석 결과:\n")
            print(result)  
            if judg == "적합":
                return {
                    "status": "success",
                    "step": step,
                    "message": result
                }      
            else :
                return {
                    "status": "branch",
                    "step": step,
                    "message": f"판단 결과: {result}",
                    "next": decision
                }          

        elif action_type == "validate":
            local_vars = step.get("context", {})  # row 값들이 dict로 들어오도록 전달해야 함
            expr = step.get("expr")
            rule_name = step.get("name", "Unnamed Validation")
            try:
                if eval(expr, {}, local_vars):  # 안전하게 평가
                    return {
                        "status": "fail",
                        "step": step,
                        "message": f"[Validation Failed] {rule_name}"
                    }
                return {
                    "status": "success",
                    "step": step,
                    "message": f"[Validation Passed] {rule_name}"
                }
            except Exception as e:
                return {
                    "status": "error",
                    "step": step,
                    "message": f"[Validation Error] {rule_name}: {e}"
                }

        # 🔒 아래 로직은 check_message가 아닐 때만 실행됨
        alert_result = handle_alert(driver)
        if alert_result:
            return {
                "status": "warning",
                "step": step,
                "message": alert_result["message"]
            }

        error_result = detect_error_message(driver)
        if error_result:
            return {
                "status": "fail",
                "step": step,
                "message": error_result["message"]
            }

        error_result = detect_nexacro_error_and_confirm(driver, wait)
        if error_result:
            return {
                "status": error_result["status"],
                "step": step,
                "message": error_result["message"]
            }

        return {
            "status": "success",
            "step": step,
            "message": "Step executed successfully"
        }

    except Exception as e:
        return {"status": "fail", "step": step, "message": str(e)}


import re

def extract_judgement(text):
    # '최종판정', '최종 판정', '판정' 등 다양한 경우를 탐지
    match = re.search(r"(최종\s*)?판정[\s]*[:：]\s*([^\n\r]+)", text)
    if not match:
        return None
    result = match.group(2).strip()
    if "부적합" in result:
        return "부적합"
    elif "적합" in result:
        return "적합"
    return None


def get_driver(headless=False):
    options = Options()
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/114.0.0.0")
    if headless:
        options.add_argument("--headless")
    return webdriver.Chrome(options=options)

def handle_alert(driver):
    try:
        alert = WebDriverWait(driver, 3).until(EC.alert_is_present())
        message = alert.text
        alert.accept()
        return {"status": "warning", "message": f"Alert detected: {message}"}
    except:
        return None
    
def detect_nexacro_error_and_confirm(driver, wait):
    try:
        # 메시지 셀 텍스트 감지
        error_cell = driver.find_element(By.ID, "mainframe.VFrameSet.frameMain.msg.freemsg.form.grdMsg.body.gridrow_0.cell_0_0:text")
        message = error_cell.text.strip()

        if message:
            print(f"⚠️ 넥사크로 메시지 감지됨: {message}")

            # 확인 버튼 클릭
            confirm_button = wait.until(EC.element_to_be_clickable((By.ID, "mainframe.VFrameSet.frameMain.msg.freemsg.form.btnOk")))
            confirm_button.click()

            return {
                "status": "fail" if '오류' in message or 'Exception' in message else 'warning',
                "message": f"Nexacro Message: {message}"
            }

    except Exception:
        pass  # 메시지가 없으면 무시
    return None


def detect_error_message(driver):
    try:
        error_element = driver.find_element(By.CSS_SELECTOR, ".error, .alert, .warning")
        if error_element.is_displayed():
            return {"status": "fail", "message": error_element.text.strip()}
    except:
        return None

def find_next_node_by_label(flow, current_node_id, label):
    for conn in flow["connections"]:
        if conn["from"] == current_node_id and conn["label"] == label:
            return conn["to"]
    return None

def select_flow_ui(flows):
    def on_submit():
        selected_idx = listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("선택 필요", "실행할 FLOW를 선택하세요.")
            return
        selected_flow_id.set(flows[selected_idx[0]][0])
        root.destroy()

    root = tk.Tk()
    root.title("FLOW 선택")
    root.geometry("400x300")
    
    tk.Label(root, text="실행할 FLOW를 선택하세요:", pady=10).pack()

    listbox = tk.Listbox(root, height=10, width=50)
    for fid, created in flows:
        listbox.insert(tk.END, f"FLOW_ID={fid} | 생성일: {created}")
    listbox.pack(pady=10)

    selected_flow_id = tk.StringVar()
    tk.Button(root, text="실행", command=on_submit).pack(pady=10)
    root.mainloop()
    
    return selected_flow_id.get()

def run_automation_by_flowid_ui(db_path, target_date):
    
    rule_map = load_db_rules(db_path)
    flows = list_flows(db_path)
    genInfo = "7284 통영천연가스CC"
    print(f"rule_map >> {rule_map}")
    print(f"flows >> {flows}")

    if not flows:
        print("❌ 사용할 수 있는 FLOW가 없습니다.")
        return

    flow_id = select_flow_ui(flows)
    if not flow_id:
        print("🚫 실행이 취소되었습니다.")
        return

    try:
        flow_data = load_flow_by_id(db_path, flow_id)
    except Exception as e:
        print(f"❌ 오류: {e}")
        return

    node_sequence = get_execution_sequence(flow_data)
    nodes = {n["id"]: n for n in flow_data["nodes"]}

    print(f"\n▶ 실행 노드 순서: {node_sequence}")

    # ✅ 시각화 초기화
    start_visualizer_thread(flow_data)
    driver = get_driver(headless=False)
    wait = WebDriverWait(driver, 15)
    actions = ActionChains(driver)
    step_results = []

    visited = set()
    next_node_override = None
    current_index = 0

    while current_index < len(node_sequence):
        if next_node_override:
            node_id = next_node_override
            next_node_override = None
        else:
            node_id = node_sequence[current_index]
            current_index += 1

        if node_id in visited:
            continue
        visited.add(node_id)

        node = nodes.get(node_id, {})
        print(f"612---NODE={node}")
        rule_id = node.get("ruleId")
        rule_name = node.get("ruleName")
        if node.get("type") == "end":
            print("🛑 End 노드 도달")
            break

        print(f"618---RULE_MAP={rule_map}")
        print(f"619---RULE_ID={rule_id}")
        print(f"620---RULE_NAME={rule_name}")
        if not rule_id or rule_id not in rule_map:
            print(f"⚠️ RULE_ID={rule_id} 를 찾을 수 없습니다.")
            continue

        rule = rule_map.get(rule_id)
        if not rule:
            print(f"⚠️ RULE_ID={rule_id} 를 찾을 수 없습니다.")
            continue

        if rule.get("url"):
            print(f"🌐 접속: {rule['url']}")
            driver.get(rule["url"])
            time.sleep(3)

        print(f"▶ 실행 중: RULE_ID={rule_id} | RULE_NAME={rule_name}")

        for step in rule.get("steps", []):
            if step.get("value") == "next_day":
                step["value"] = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")

            result = process_step(driver, wait, actions, step, target_date,  genInfo )
            result["nodeId"] = rule_name  # 이름 기반으로 표시
            step_results.append(result)

            # ✅ 노드 상태 업데이트 + 라벨 유지
            icon = "✅" if result["status"] == "success" else "❌" if result["status"] == "fail" else "⚠️"
            update_node_status_safe(node_id, result["status"], rule_name )

            _visual_root.update_idletasks()
            _visual_root.update()

            if result["status"] == "branch" and result.get("next"):
                next_label = result["next"]
                next_node_id = find_next_node_by_label(flow_data, node_id, next_label)
                if next_node_id:
                    print(f"🔀 분기 이동: {node_id} ➝ {next_node_id} (label: {next_label})")
                    next_node_override = next_node_id
                    break

            elif result["status"] == "fail":
                print(f"❌ Step 실패: {result['step'].get('label', '')} - {result['message']}")

    print("\n📋 실행 결과 요약:")
    for i, result in enumerate(step_results, 1):
        status_icon = "✅" if result["status"] == "success" else "❌"
        label = result["step"].get("label", "") or result["step"].get("selector", "")
        print(f"{i:02d}. {status_icon} {label} - {result['message']}")

    print("\n✅ 전체 작업 완료. Enter 키를 누르면 종료됩니다.")
    #input()
    driver.quit()
    
def run_automation_by_flowid(db_path):
    rule_map = load_db_rules(db_path)
    flows = list_flows(db_path)

    print("\n🧩 선택 가능한 FLOW 목록:")
    for fid, created in flows:
        print(f"  - FLOW_ID={fid} | 생성일: {created}")

    flow_id = input("\n실행할 FLOW_ID를 입력하세요: ").strip()
    try:
        flow_data = load_flow_by_id(db_path, flow_id)
    except Exception as e:
        print(f"❌ 오류: {e}")
        return

    node_sequence = get_execution_sequence(flow_data)
    nodes = {n["id"]: n for n in flow_data["nodes"]}

    print(f"\n▶ 실행 노드 순서: {node_sequence}")

    driver = get_driver(headless=False)
    wait = WebDriverWait(driver, 15)
    actions = ActionChains(driver)
    step_results = []

    visited = set()
    next_node_override = None
    current_index = 0

    while current_index < len(node_sequence):
        # 현재 노드 ID 결정
        if next_node_override:
            node_id = next_node_override
            next_node_override = None
        else:
            node_id = node_sequence[current_index]
            current_index += 1  # next_node_override인 경우엔 증가시키지 않음!

        if node_id in visited:
            continue
        visited.add(node_id)

        node = nodes.get(node_id, {})
        rule_id = node.get("ruleId")
        rule_name = node.get("ruleName")
        # END 노드는 실행 생략
        if node.get("type") == "end":
            print("🛑 End 노드 도달")
            break

        # ruleId 없는 경우도 생략
        if not rule_id or rule_id not in rule_map:
            print(f"⚠️ RULE_ID={rule_id} 를 찾을 수 없습니다.")
            continue

        rule = rule_map.get(rule_id)

        if not rule:
            print(f"⚠️ RULE_ID={rule_id} 를 찾을 수 없습니다.")
            continue

        # URL 있으면 접속
        if rule.get("url"):
            print(f"🌐 접속: {rule['url']}")
            driver.get(rule["url"])
            time.sleep(3)

        print(f"▶ 실행 중: RULE_ID={rule_id} | RULE_NAME={rule_name}")

        for step in rule.get("steps", []):
            if step.get("value") == "next_day":
                step["value"] = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            elif step.get("value") == "" or  step.get("inputdt"):
                step["value"] = target_date

            result = process_step(driver, wait, actions, step , target_date, genInfo )
            step_results.append(result)

            if result["status"] == "branch" and result.get("next"):
                next_label = result["next"]
                next_node_id = find_next_node_by_label(flow_data, node_id, next_label)
                if next_node_id:
                    print(f"🔀 분기 이동: {node_id} ➝ {next_node_id} (label: {next_label})")
                    next_node_override = next_node_id
                    break  # 다음 노드로 이동

            elif result["status"] == "fail":
                print(f"❌ Step 실패: {result['step'].get('label', '')} - {result['message']}")
                # 중단 없이 다음 step으로 진행

    print("\n📋 실행 결과 요약:")
    for i, result in enumerate(step_results, 1):
        status_icon = "✅" if result["status"] == "success" else "❌"
        label = result["step"].get("label", "") or result["step"].get("selector", "")
        print(f"{i:02d}. {status_icon} {label} - {result['message']}")

    print("\n✅ 전체 작업 완료. Enter 키를 누르면 종료됩니다.")
    draw_execution_result(flow_data, step_results)
    input()
    driver.quit()
# 아래 함수 수정:
def run_automation_by_flowid_ai(db_path, flow_id, vgenInfo, tradeYmd, log_callback=None, session_id=None):
    import time
    from datetime import datetime, timedelta
    import Flow_Visualizer  # 재귀 호출 방지용
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.common.action_chains import ActionChains

    # 내부에서 메시지를 처리할 log 함수
    def sse_log(msg):
        if log_callback:
            log_callback(msg)

    target_date = tradeYmd
    if vgenInfo :
        genInfo = vgenInfo

    rule_map = Flow_Visualizer.load_db_rules(db_path)
    flows = Flow_Visualizer.list_flows(db_path)
    
    try:
        flow_data = Flow_Visualizer.load_flow_by_id(db_path, flow_id)
    except Exception as e:
        err = f"❌ 오류: {e}"
        sse_log(err)
        return

    node_sequence = Flow_Visualizer.get_execution_sequence(flow_data)
    nodes = {n["id"]: n for n in flow_data["nodes"]}

    sse_log(f"\n▶ 실행 노드 순서: {node_sequence}")

    # ✅ 시각화 초기화
    Flow_Visualizer.start_visualizer_thread(flow_data)
    driver = Flow_Visualizer.get_driver(headless=False)
    wait = WebDriverWait(driver, 15)
    actions = ActionChains(driver)
    step_results = []

    visited = set()
    next_node_override = None
    current_index = 0

    while current_index < len(node_sequence):
        if next_node_override:
            node_id = next_node_override
            next_node_override = None
        else:
            node_id = node_sequence[current_index]
            current_index += 1

        if node_id in visited:
            continue
        visited.add(node_id)

        node = nodes.get(node_id, {})
        rule_id = node.get("ruleId")
        rule_name = node.get("ruleName")
        if node.get("type") == "end":
            sse_log("🛑 End 노드 도달")
            break

        if not rule_id or rule_id not in rule_map:
            sse_log(f"⚠️ RULE_ID={rule_id} 를 찾을 수 없습니다.")
            continue

        rule = rule_map.get(rule_id)
        if not rule:
            sse_log(f"⚠️ RULE_ID={rule_id} 를 찾을 수 없습니다.")
            continue

        if rule.get("url"):
            msg = f"🌐 접속: {rule['url']}"
            sse_log(msg)
            driver.get(rule["url"])
            time.sleep(3)

        msg = f"▶ 실행 중: RULE_ID={rule_id} | RULE_NAME={rule_name}"
        sse_log(msg)

        for step in rule.get("steps", []):
            # 날짜 처리
            if step.get("value") == "next_day":
                step["value"] = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
            elif step.get("value") == "" or step.get("value") == "inputdt":
                step["value"] = target_date

            result = process_step(driver, wait, actions, step, target_date, genInfo )
            result["nodeId"] = rule_name  # 이름 기반으로 표시

            node_id2 = result.get("nodeId")
            status = result.get("status", "unknown")
            message = result.get("message", "")  # 에러 메시지 등

            # DB 로그 기록
            Flow_Visualizer.log_node_execution(flow_id, node_id2, status, message, db_path)
            # 노드 상태 시각화 업데이트
            Flow_Visualizer.update_node_status_safe(node_id, status, rule_name)
            if Flow_Visualizer._visual_root:
                Flow_Visualizer._visual_root.update_idletasks()
                Flow_Visualizer._visual_root.update()

            # 단계별 로그 전송
            label = step.get('label', step.get('selector', ''))
            log_msg = f"  - {label}: {status} ({message})"
            sse_log(log_msg)

            step_results.append(result)

            # 분기 처리
            if result["status"] == "branch" and result.get("next"):
                next_label = result["next"]
                next_node_id = Flow_Visualizer.find_next_node_by_label(flow_data, node_id, next_label)
                if next_node_id:
                    sse_log(f"🔀 분기 이동: {node_id} ➝ {next_node_id} (label: {next_label})")
                    next_node_override = next_node_id
                    break

            elif result["status"] == "fail":
                sse_log(f"❌ Step 실패: {label} - {message}")

    # 전체 결과 요약
    sse_log("\n📋 실행 결과 요약:")
    for i, result in enumerate(step_results, 1):
        status_icon = "✅" if result["status"] == "success" else "❌"
        label = result["step"].get("label", "") or result["step"].get("selector", "")
        sse_log(f"{i:02d}. {status_icon} {label} - {result['message']}")

    sse_log("\n✅ 전체 작업 완료.")
    if session_id:
        # 만약 SSE용 큐가 있다면 종료 신호
        try:
            q = get_log_queue(session_id)
            q.put("[[FLOW_END]]")
        except Exception:
            pass

    driver.quit()
    return step_results

def log_collector(msg, session_id):
    print("[SSE LOG]", msg)  # <--- 콘솔로도 찍힘
    q = get_log_queue(session_id)
    q.put(msg)

def log_node_execution(flow_id, node_id, status, message, db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO WEBRULE_EXEC_LOG (flow_id, node_id, status, message)
                VALUES (?, ?, ?, ?)
            """, (flow_id, node_id, status, message))
            conn.commit()
    except Exception as e:
        print(f"❌ 로그 저장 실패: {e}")

# utils.py
import queue

log_stream_queues = {}  # session_id: queue.Queue()

def get_log_queue(session_id):
    if session_id not in log_stream_queues:
        log_stream_queues[session_id] = queue.Queue()
    return log_stream_queues[session_id]        
