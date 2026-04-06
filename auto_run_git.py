import subprocess
import time
import os

# ==========================================
# 設定エリア:
# ==========================================
patterns = [
    "AAAABB",
    "AAABAB",
    "AAABBA",
    "AABAAB",
    "AABABA",
    "AABBAA",
    "ABAAAB",
    "ABAABA",
    "ABABAA",
    "ABBAAA",
    "BAAAAB",
    "BAAABA",
    "BAABAA",
    "BABAAA",
    "BBAAAA"
]

# 各パターンごとの試行回数 (本番は5回)
RUNS_PER_PATTERN = 5

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
    print(f" Start FINAL Auto Optimization (With Images)")
    print(f" Patterns: {len(patterns)}")
    print(f" Runs/Pattern: {RUNS_PER_PATTERN}")
    print(f" Total Runs: {total_tasks}")
    print(f"==================================================")

    for pattern in patterns:
        print(f"\n>>>>>>>>>> Starting Pattern: {pattern} <<<<<<<<<<")
        
        for i in range(RUNS_PER_PATTERN):
            current_task += 1
            run_id = i + 1
            print(f"\n--- Progress: {current_task}/{total_tasks} (Pattern: {pattern}, Run: {run_id}/{RUNS_PER_PATTERN}) ---")

            # 1. main.py を実行
            # 画像を生成するため --skip-plot は付けません
            success = run_command(["python", "main.py", "--pattern", pattern])
            
            if not success:
                print("Optimization failed. Continuing to next run...")
                continue

            # 2. Git Commit & Push
            # resultフォルダを強制的に追加 (-f)
            run_command(["git", "add", "-f", "result"])
            
            # コミットメッセージ: 「ABBAABの配置」 (Run 1)
            commit_message = f"「{pattern}の配置」 (Run {run_id})"
            
            # コミット実行
            run_command(["git", "commit", "-m", commit_message])
            
            # プッシュ実行
            push_success = run_command(["git", "push"])
            
            if not push_success:
                print("[Warning] Git push failed. (Data is saved locally)")

            # GitHubへの負荷軽減のため少し待機
            time.sleep(2)

    print("\n==================================================")
    print(" All tasks completed successfully!")
    print("==================================================")
    
    # ==========================================
    # EC2の自動停止処理
    # ==========================================
    print("\n[System] Shutting down EC2 instance in 1 minute...")
    time.sleep(60) 
    
    # シャットダウン実行
    subprocess.run(["sudo", "shutdown", "-h", "now"])

if __name__ == "__main__":

    main()

