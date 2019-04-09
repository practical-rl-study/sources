# sources
source codes


## Installation
- [OpenAI gym](https://gym.openai.com/) 으로 놀아보기
```python
pip install -r requirement.txt
```
- Quick start  
```python
import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()
```

## Troubleshooting
- msvcp140.dll 에러
    ```python
    ImportError: Could not find 'msvcp140.dll'. TensorFlow requires that this DLL be installed in a directory that is named in your %PATH% environment variable. You may install this DLL by downloading Visual C++ 2015 Redistributable Update 3 from this URL:https://www.microsoft.com/en-us/download/details.aspx?id=53587
    ```
    - [Visual Studio 2015용 Visual C++ 재배포 가능 패키지](https://www.microsoft.com/ko-kr/download/details.aspx?id=48145) 설치

- atari-py window 설치 오류
    ```python
    $ pip install atari_py
    ....

    copying atari_py\atari_roms\zaxxon.bin -> build\lib.win-amd64-3.6\atari_py\atari_roms
    copying atari_py\package_data.txt -> build\lib.win-amd64-3.6\atari_py
    running build_ext
    Unable to execute 'make build -C atari_py/ale_interface -j 7'. HINT: are you sure `make` is installed?
    error: [WinError 2] 지정된 파일을 찾을 수 없습니다
    ```

    - 방법1
    ```python
    $ pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
    ```

    - 아래와 같이 <b>'certificate verify failed'</b> 실패시 방법2
    ```python
    Looking in links: https://github.com/Kojoley/atari-py/releases
    Collecting atari_py
    Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)'),)': /Kojoley/atari-py/releases
    Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)'),)': /Kojoley/atari-py/releases
    Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)'),)': /Kojoley/atari-py/releases
    Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)'),)': /Kojoley/atari-py/releases
    Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)'),)': /Kojoley/atari-py/releases
    Could not fetch URL https://github.com/Kojoley/atari-py/releases: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded with url: /Kojoley/atari-py/releases (Caused by SSLError(SSLError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)'),)) - skipping
    Could not find a version that satisfies the requirement atari_py (from versions: )
    No matching distribution found for atari_py
    ```
    - 방법2  
    
    %PYTHON_HOME%\Lib\site-packages\pip\_vendor\requests\sessions.py  
    ```self.verify = True``` 를 ```False``` 로 수정
    ```python
    # sessions.py  

    ...
    #: SSL Verification default.
    self.verify = False
    ...
    ```
    
    - 다시 설치 시작  
    
    ```python
    $ pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
    ```
    ```python
    Looking in links: https://github.com/Kojoley/atari-py/releases
    Collecting atari_py
    Downloading https://github.com/Kojoley/atari-py/releases/download/0.1.7/atari_py-0.1.7-cp36-cp36m-win_amd64.whl (673kB)
        100% |████████████████████████████████| 675kB 204kB/s
    Requirement already satisfied: numpy in c:\00.development\repository\github\sources\.venv\lib\site-packages (from atari_py) (1.16.2)
    Requirement already satisfied: six in c:\00.development\repository\github\sources\.venv\lib\site-packages (from atari_py) (1.12.0)
    Installing collected packages: atari-py
    Successfully installed atari-py-0.1.7
    ```
    - Test  
    ```python
    import gym

    env =gym.make("SpaceInvaders-v0")
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())
    env.env.close()
    ```

