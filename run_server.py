import platform
import subprocess
import time
import webbrowser


def run_command(command):
    system = platform.system().lower()
    if system == "darwin":  # macOS
        script = f'tell app "Terminal" to do script "{command}"'
        subprocess.run(["osascript", "-e", script])
    elif system == "linux":  # Linux
        script = f"{command}; exec bash"
        subprocess.run(["gnome-terminal", "--", "bash", "-c", script])
    elif system == "windows":  # Windows
        subprocess.Popen(["start", "cmd", "/k", command], shell=True)


def run_commands_in_single_terminal(commands):
    concatenated_commands = " && ".join(commands)
    run_command(concatenated_commands)


def main():
    run_command("mlflow server --host 127.0.0.1 --port 5000")
    time.sleep(5)

    commands_to_run = [
        "poetry install",
        "pre-commit run -a",
        "python3 bank_personal_loan_modelling/train.py",
        "python3 bank_personal_loan_modelling/infer.py",
    ]

    try:
        run_commands_in_single_terminal(commands_to_run)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed: {e}")

    time.sleep((len(commands_to_run) + 1) * 5)
    webbrowser.open("http://localhost:5000")


if __name__ == "__main__":
    main()
