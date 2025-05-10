import subprocess 


def process():
    processes = []
    num_files = 599
    for i in range(10):
        start = i * num_files // 10
        end = (i + 1) * num_files // 10
        cmd = [
            "python", "reformat.py",
            "--start_index", str(start),
            "--end_index", str(end),
        ]
        processes.append(subprocess.Popen(cmd))
    

    for p in processes:
        p.wait()


if __name__ == "__main__":
    process()