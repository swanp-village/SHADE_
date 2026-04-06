import subprocess
import time
import os

# ==========================================
# テスト用設定: No.8 (ABBAAB) を1回だけ
# ==========================================
patterns = [
    "ABBAAB"  # No.8のみ
]

# 試行回数は1回だけ
RUNS_PER_PATTERN = 1

# ==========================================
# 自動実行ロジック
# ==========================================

def run_command(command_list):
    """コマンドを実行し、ログを表示する関数"""
    cmd_str = " ".join(command_list)
    print(f"\n[System] Executing: {cmd_str}")
    try:
        # subprocess.runでコマンドを実行（完了するまで待機）
        result = subprocess.run(command_list, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed: {cmd_str}")
        print(e)
        return False

def main():
    total_tasks = len(patterns) * RUNS_PER_PATTERN
    current_task = 0

    print(f"==================================================")
    print(f" Start TEST Run with Auto Shutdown")
    print(f" Patterns: {patterns}")
    print(f" Runs/Pattern: {RUNS_PER_PATTERN}")
    print(f"==================================================")

    for pattern in patterns:
        print(f"\n>>>>>>>>>> Starting Pattern: {pattern} <<<<<<<<<<")
        
        for i in range(RUNS_PER_PATTERN):
            current_task += 1
            run_id = i + 1
            print(f"\n--- Progress: {current_task}/{total_tasks} (Pattern: {pattern}, Run: {run_id}/{RUNS_PER_PATTERN}) ---")

            # 1. main.py を実行 (--skip-plotあり)
            success = run_command(["python", "main.py", "--pattern", pattern, "--skip-plot"])
            
            if not success:
                print("Optimization failed.")
                # テストなのでエラーが出たら止める場合はここで break しても良いですが、
                # シャットダウンのテストも兼ねているので続行します

            # 2. Git Commit & Push
            run_command(["git", "add", "-f", "result"])
            
            # テスト用のコミットメッセージ
            commit_message = f"TEST: Optimization result for {pattern} (Run {run_id})"
            
            run_command(["git", "commit", "-m", commit_message])
            push_success = run_command(["git", "push"])
            
            if not push_success:
                print("[Warning] Git push failed.")

            time.sleep(2)

    print("\n==================================================")
    print(" Test run completed successfully!")
    print("==================================================")
    
    # ==========================================
    # EC2の自動停止処理 (テスト)
    # ==========================================
    print("\n[System] Shutting down EC2 instance in 1 minute...")
    print("If you want to CANCEL shutdown, press Ctrl+C now!")
    
    # 60秒待機（この間にログなどが書き込まれます）
    time.sleep(60) 
    
    # シャットダウン実行
    print("Executing shutdown command...")
    subprocess.run(["sudo", "shutdown", "-h", "now"])

if __name__ == "__main__":
    main()