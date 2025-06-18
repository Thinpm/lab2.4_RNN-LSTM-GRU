import subprocess
import time

def run_experiment(script_name):
    print(f"\nĐang chạy thử nghiệm với {script_name}...")
    start_time = time.time()
    process = subprocess.run(['python', script_name], capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Thời gian chạy: {end_time - start_time:.2f} giây")
    print("Output:")
    print(process.stdout)
    
    if process.stderr:
        print("Errors:")
        print(process.stderr)
    
    return end_time - start_time

def main():
    experiments = [
        'image_captioning_improved.py',  # LSTM
        'image_captioning_rnn.py',       # RNN
        'image_captioning_gru.py'        # GRU
    ]
    
    results = {}
    for script in experiments:
        runtime = run_experiment(script)
        results[script] = runtime
    
    print("\nKết quả tổng hợp:")
    print("-" * 50)
    for script, runtime in results.items():
        print(f"{script}: {runtime:.2f} giây")

if __name__ == "__main__":
    main() 