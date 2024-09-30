# TopoTrafficRL

Implementation of reinforcement learning based dense traffic awareness method.

## Install

1. Dependencies
    ```sh
    pip3 install gymnasium==0.29.1
    pip3 install numpy==1.26.4
    pip3 install moviepy imageio_ffmpeg tensorboard tensorboardx pyvirtualdisplay IPython
    sudo apt-get install -y xvfb ffmpeg
    ```
2. Install TopoTrafficRL
    ```sh
    pip3 install .
    ```

## Usage

1. Run the example
    ```sh
    python3 scripts/example.py
    ```

## License
- The environment design is based on HighwayEnv by Edouard Leurent. The original github link is [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).
- The agent design is based on rl-agents by Edouard Leurent as well. The original github link is [rl-agents](https://github.com/eleurent/rl-agents).
- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
