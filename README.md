# PPO_AI
AI model utilizing the Proximal Policy Optimization algorithm in the common Actor Critic style network strategy. This branch in particular utilizes the atari 128 byte ram environments from OpenAI.

### Installation
Should be able to get working with pip package manager. Simply type "pip install -r requirements.txt"
from the main dir. 

Microsoft Visual Studio 2015 Build Tools is Needed for the NES_PY module that is used to run environments I utilize. Download from [here.](https://www.microsoft.com/en-us/download/details.aspx?id=48159)

Additionally, make sure to edit "Main.py" for the approproate number of environments to run. Every environment is
run in it's own thread. Command line args will be added in the future.
