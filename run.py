import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=cwd)
    return result.stdout, result.stderr


if __name__ == "__main__":
    commands = [
        "cmake -DCMAKE_BUILD_TYPE=Release ..",
        "cmake --build . -- -j4"
    ]
    for cmd in commands:
        run_command(cmd, cwd='cmake-build-release')

    commands = [
        "./cmake-build-release/example_search 32 200 70 200",
        "./cmake-build-release/example_search 32 200 800000 200",
        "./cmake-build-release/example_search 32 200 3200000 200",
        "./cmake-build-release/example_search 32 200 4800000 200",
    ]

    with ProcessPoolExecutor() as executor:
        # 提交所有命令并行执行
        futures = {executor.submit(run_command, cmd): cmd for cmd in commands}

        for future in as_completed(futures):
            cmd = futures[future]
            try:
                stdout, stderr = future.result()
                print(f"Output of {cmd}:\n{stdout}")
                if stderr:
                    print(f"Error output of {cmd}:\n{stderr}")
                print("====================================")
            except Exception as e:
                print(f"{cmd} generated an exception: {e}")
