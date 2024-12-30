import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import datetime
import typer
import requests


assert os.getenv("SENDER_PASSWORD") is not None, "SENDER_PASSWORD is not set"
assert os.getenv("RECIPIENT_EMAIL") is not None, "RECIPIENT_EMAIL is not set"
assert os.getenv("SENDER_EMAIL") is not None, "SENDER_EMAIL is not set"

def contain_error(line: str) -> bool:
    if "200 OK" in line:
        return False
    return True


def send_email(error_type: str, message: str):
    SMTP_SERVER = "smtp.gmail.com"  # Replace with your SMTP server
    SMTP_PORT = 587  # Commonly used port for TLS
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

    # Compose Email
    subject = f"Error Found: {error_type}"
    now = datetime.datetime.now()
    body = f"Time: {now}\nContent of error: {message}"

    message = MIMEMultipart()
    message["From"] = SENDER_EMAIL
    message["To"] = RECIPIENT_EMAIL
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Send Email
    try:
        # Connect to the SMTP server
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Upgrade connection to secure TLS
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, message.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def check_health(port: int):
    try:
        url = f"http://127.0.0.1:{port}/health"
        response = requests.get(url)
        return response.status_code == 200
    except Exception as e:
        print(f"Error checking health: {e}")
        return False


def monitor_log_file(file_path: str, interval: int = 5, port: int = 8000):
    """
    Monitors a log file to check if new requests are being added and if there are any errors.

    Args:
        file_path (str): Path to the log file.
        interval (int): Time interval (in seconds) to check for updates.

    Returns:
        None
    """
    try:
        print(f"Monitoring log file: {file_path}")
        if not os.path.exists(file_path):
            print("Error: Log file does not exist.")
            return

        last_size = os.path.getsize(file_path)

        while True:
            time.sleep(interval)
            print("--------------------------------")
            if check_health(port):
                print("Service is healthy.")
            else:
                print("Service is unhealthy.")
                send_email("Service is unhealthy.", "Service is unhealthy.")

            
            current_size = os.path.getsize(file_path)

            if current_size > last_size:
                with open(file_path, "r") as log_file:
                    log_file.seek(last_size)  # Start reading from the last checked position
                    new_lines = log_file.readlines()

                contain_new_requests = False
                for line in new_lines:
                    temp_line = line.strip()
                    if "GET /health HTTP/1.1" in temp_line:
                        continue # ignore health check requests
                    
                    if contain_error(temp_line):
                        send_email("Error detected in Request", temp_line)
                        print(f"Error detected: {temp_line}")
                    if "POST /v1/chat/completions" in temp_line:
                        contain_new_requests = True
                        
                if not contain_new_requests:
                    print("No new requests detected.")
                    send_email("No new requests detected.", "No more requests detected.")
                else:
                    print("New requests detected.")

                last_size = current_size
            elif current_size < last_size:
                last_size = current_size # maybe the file is being truncated or overwritten
            elif current_size == last_size:
                print("No new requests detected.")
                send_email("File is unchanged", "File is unchanged")
            
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")

# Example usage
if __name__ == "__main__":
    typer.run(monitor_log_file)
