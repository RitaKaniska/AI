import requests

# Thông tin đăng nhập
username = "team32"
password = "zxD5KUfPvh"

# Step 1: Login
def login():
    login_url = "https://eventretrieval.one/api/v2/login"
    login_data = {
        "username": username,
        "password": password
    }
    
    response = requests.post(login_url, json=login_data)
    
    if response.status_code == 200:
        session_id = response.json().get("sessionid")
        print("Login successful. Session ID:", session_id)
        return session_id
    else:
        print("Login failed:", response.status_code, response.text)
        return None

# Step 2: Get EvaluationID
def get_evaluation_id(session_id):
    eval_url = f"https://eventretrieval.one/api/v2/client/evaluation/list?session={session_id}"
    
    response = requests.get(eval_url)
    
    if response.status_code == 200:
        evaluation_id = response.json()[0].get("evaluationId")  # Giả sử lấy ID đầu tiên
        print("Evaluation ID:", evaluation_id)
        return evaluation_id
    else:
        print("Failed to get Evaluation ID:", response.status_code, response.text)
        return None

# Step 3.2: Submit SOL (QA)
def submit_sol(session_id, evaluation_id):
    submit_url = f"https://eventretrieval.one/api/v2/submit/{evaluation_id}?session={session_id}"
    
    submit_data = {
        "answerSets": [
            {
                "answers": [
                    {"text": "3"}  # Điền kết quả của bạn tại đây
                ]
            }
        ]
    }
    
    response = requests.post(submit_url, json=submit_data)
    
    if response.status_code == 200:
        print("Submit successful:", response.json())
    else:
        print("Submit failed:", response.status_code, response.text)

# Main function
if __name__ == "__main__":
    session_id = login()
    if session_id:
        evaluation_id = get_evaluation_id(session_id)
        if evaluation_id:
            submit_sol(session_id, evaluation_id)
