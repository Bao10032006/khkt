import os
import webbrowser
import threading

def excuteCMD():

    folder_path = r'E:/2025/khkt/di thi/test_1'
    os.chdir(folder_path)

    # Command to be executed
    command = 'node server.js'  # Example command

    # Running the command'
    
    os.system(command)
    print("3")
    

def openLink():
    # Đường dẫn đến trình duyệt (thay đổi theo đường dẫn của trình duyệt trên hệ thống của bạn)
    chrome_path = r'C:/Program Files/Google/Chrome/Application/chrome.exe'  # Đường dẫn đến Chrome
    webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))

    # Liên kết bạn muốn mở
    url = 'http://localhost:3000'

    
    # Mở liên kết trong Chrome
    webbrowser.get('chrome').open(url)

thread1 = threading.Thread(target=excuteCMD)
thread2 = threading.Thread(target=openLink)

# Khởi động các luồng
thread1.start()
# thread2.start()

# Chờ cho các luồng hoàn thành
