import time 


def append_to_file(file_path: str, content: str):
    import os
    
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            pass
        
    with open(file_path, "a") as f:
        f.write(content + "\n")


def main():
    count = 0
    while True:
        append_to_file("logs/mock_miner.log", f"Hello, world! {count}")
        content = """INFO:     83.143.115.156:62342 - "POST /QueryTask HTTP/1.1" 200 OK"""
        error_content = """INFO:     83.143.115.156:62342 - "POST /QueryTask HTTP/1.1" 500 Internal Server Error"""
        if count % 1000 == 0 and count > 0:
            append_to_file("logs/mock_miner.log", error_content)
        else:
            append_to_file("logs/mock_miner.log", content)
        count += 1
        time.sleep(2)


if __name__ == "__main__":
    main()
