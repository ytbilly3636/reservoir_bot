# reservoir_bot

This repository is for a project entitled "Natural Language Processing by Reservoir Computing for AI-bot on Jetson," which is for an application of Jetson AI Specialist.

YouTube video of this project is [available here](https://www.youtube.com/watch?v=FEEE70qeIC8).


## Setup
The code works on a Jetson Nano 4GB with JetPack 4.6.1.  

### Step 1: Create a bot
1. Create your discord account and a server where a bot is used
2. Access [Discord Develper Portal](https://discord.com/developers/applications)
3. Click "New Application" to create an application (any name is OK)
4. Click "Bot" in Settings
    1. Click "Reset Token" in Biild-A-Bot and copy the token
    2. Check "MESSAGE CONTENT INTENT" in Privileged Gateway Intents
5. Click OAuth2 -> URL Generator
    1. Check "bot" in SCOPES
    2. Check "Manage Message" and "Read Message History" in BOT PERMISSIONS
    3. Copy Generated URL and access it to add the bot to the server

### Step 2: Create a Docker environment
```
$ git clone https://github.com/ytbilly3636/reservoir_bot.git
$ cd reservoir_bot
$ docker build -t env .
```

If you do not want to use Docker, install the following packages:
- cupy-cuda102
- discord.py==1.7.3
- mecab-python3
- unidic-lite
- gensim

### Step 3: Download a word2vec model
1. Access [shiroyagicorp/japanese-word2vec-model-builder](https://github.com/shiroyagicorp/japanese-word2vec-model-builder)
2. Download the trained model and unzip it in `reservoir_bot/`

### Step 4: Create a textfile including the token
1. Create `token.txt` in `reservoir_bot/`
2. Paste the copied token in `token.txt`


## Run
```
$ ./run.sh
```

if you do not want to use Docker, execute `main.py`.
```
$ python main.py
```