# download_tracking.py
from SoccerNet.Downloader import SoccerNetDownloader

# change this to your target folder
local_dir = r"C:\Data\SoccerNet"   # or "/home/you/SoccerNet"

myDownloader = SoccerNetDownloader(LocalDirectory=local_dir)

# download only the tracking task (12 games + clips + annotations)
myDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])
# If you want 2023 tracking additions:
# myDownloader.downloadDataTask(task="tracking-2023", split=["train","test","challenge"])
